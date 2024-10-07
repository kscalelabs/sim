""" Calibrate the joint position of the robot.

Run:
    python sim/scripts/calibration_mujoco.py --embodiment stompypro
"""
import argparse
import math
import os
from collections import deque
from copy import deepcopy

import matplotlib.pyplot as plt  # Add this import for plotting


import mujoco
import mujoco_viewer
import numpy as np
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

from sim.scripts.create_mjcf import load_embodiment

import torch  # isort: skip


class cmd:
    vx = 0.5
    vy = 0.0
    dyaw = 0.0


def quaternion_to_euler_array(quat):
    # Ensure quaternion is in the correct format [x, y, z, w]
    x, y, z, w = quat

    # Roll (x-axis rotation)
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = np.arctan2(t0, t1)

    # Pitch (y-axis rotation)
    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch_y = np.arcsin(t2)

    # Yaw (z-axis rotation)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)

    # Returns roll, pitch, yaw in a NumPy array in radians
    return np.array([roll_x, pitch_y, yaw_z])


def get_obs(data):
    """Extracts an observation from the mujoco data structure"""
    q = data.qpos.astype(np.double)
    dq = data.qvel.astype(np.double)
    quat = data.sensor("orientation").data[[1, 2, 3, 0]].astype(np.double)
    r = R.from_quat(quat)
    v = r.apply(data.qvel[:3], inverse=True).astype(np.double)  # In the base frame
    omega = data.sensor("angular-velocity").data.astype(np.double)
    gvec = r.apply(np.array([0.0, 0.0, -1.0]), inverse=True).astype(np.double)
    return (q, dq, quat, v, omega, gvec)


def pd_control(target_q, q, kp, target_dq, dq, kd, default):
    """Calculates torques from position commands"""
    return kp * (target_q + default - q) - kd * dq


def run_mujoco(cfg, joint_id=0, steps=1000):
    """
    Run the Mujoco motor identification.

    Args:
        cfg: The configuration object containing simulation settings.

    Returns:
        None
    """
    model_dir = os.environ.get("MODEL_DIR")
    mujoco_model_path = f"{model_dir}/{args.embodiment}/robot_calibration.xml"
    model = mujoco.MjModel.from_xml_path(mujoco_model_path)
    model.opt.timestep = cfg.sim_config.dt
    data = mujoco.MjData(model)
    joint_names = [model.joint(joint_id).name for joint_id in range(model.njnt)]
    print("Joint names:", joint_names)

    try:
        data.qpos = model.keyframe("default").qpos
        default = deepcopy(model.keyframe("default").qpos)[-cfg.num_actions :]
        print("Default position:", default)
    except:
        print("No default position found, using zero initialization")
        default = np.zeros(cfg.num_actions)  # 3 for pos, 4 for quat, cfg.num_actions for joints

    mujoco.mj_step(model, data)

    data.qvel = np.zeros_like(data.qvel)
    data.qacc = np.zeros_like(data.qacc)
    viewer = mujoco_viewer.MujocoViewer(model, data)

    target_q = np.zeros((cfg.num_actions), dtype=np.double)

    # Parameters for calibration
    step = 0
    dt = cfg.sim_config.dt
    joint_id = 0
    # Lists to store time and position data for plotting
    time_data = []
    position_data = []

    while step < steps:
        q, dq, quat, v, omega, gvec = get_obs(data)
        q = q[-cfg.num_actions :]
        dq = dq[-cfg.num_actions :]

        sin_pos = torch.tensor(math.sin(2 * math.pi * step * dt / cfg.sim_config.cycle_time))
        
        # N hz control loop
        if step % cfg.sim_config.decimation == 0:
            target_q = default
            target_q[joint_id] = sin_pos

        time_data.append(step)
        position_data.append(q - default)

        # Generate PD control
        tau = pd_control(
            target_q, q, cfg.robot_config.kps, default, dq, cfg.robot_config.kds, default
        )  # Calc torques

        tau = np.clip(tau, -cfg.robot_config.tau_limit, cfg.robot_config.tau_limit)  # Clamp torques
        
        data.ctrl = tau

        mujoco.mj_step(model, data)
        step += 1
        viewer.render()
        
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(time_data, position_data)
    plt.title("Joint Position over Time")
    plt.xlabel("Step (10hz)")
    plt.ylabel("Joint Position")
    plt.grid(True)
    plt.savefig("joint_position_plot.png")
    plt.show()

    viewer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deployment script.")
    parser.add_argument("--embodiment", type=str, required=True, help="embodiment")
    parser.add_argument("--joint_id", type=int, default=1, help="joint_id")
    args = parser.parse_args()

    robot = load_embodiment(args.embodiment)

    class Sim2simCfg:
        num_actions = len(robot.all_joints())

        class env:
            num_actions = len(robot.all_joints())
            frame_stack = 15
            c_frame_stack = 3
            num_single_obs = 11 + num_actions * c_frame_stack
            num_observations = int(frame_stack * num_single_obs)

        class sim_config:
            sim_duration = 60.0
            dt = 0.001
            decimation = 20
            cycle_time = 0.64

        class robot_config:
            tau_factor = 0.85
            tau_limit = np.array(list(robot.stiffness().values()) + list(robot.stiffness().values())) * tau_factor
            kps = tau_limit
            kds = np.array(list(robot.damping().values()) + list(robot.damping().values()))

        class normalization:
            class obs_scales:
                lin_vel = 2.0
                ang_vel = 1.0
                dof_pos = 1.0
                dof_vel = 0.05

            clip_observations = 18.0
            clip_actions = 18.0

        class control:
            action_scale = 0.25

    run_mujoco(Sim2simCfg(), joint_id=0)
