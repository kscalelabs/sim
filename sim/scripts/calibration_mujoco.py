"""Calibrate the joint position of the robot.

Run:
    python sim/scripts/calibration_mujoco.py --embodiment stompypro
"""

import argparse
import math
import os
from copy import deepcopy
from typing import Any

import matplotlib.pyplot as plt  # Add this import for plotting
import mujoco
import mujoco_viewer
import numpy as np
from scipy.spatial.transform import Rotation as R

from sim.scripts.create_mjcf import load_embodiment
from csv import writer

import torch  # isort: skip


def get_obs(data: mujoco.MjData) -> tuple:
    """Extracts an observation from the mujoco data structure.

    Args:
        data: The mujoco data structure.

    Returns:
        A tuple containing the joint positions, velocities,
        quaternions, linear velocities, angular velocities, and gravity vector.
    """
    q = data.qpos.astype(np.double)
    dq = data.qvel.astype(np.double)
    quat = data.sensor("orientation").data[[1, 2, 3, 0]].astype(np.double)
    r = R.from_quat(quat)
    v = r.apply(data.qvel[:3], inverse=True).astype(np.double)  # In the base frame
    omega = data.sensor("angular-velocity").data.astype(np.double)
    gvec = r.apply(np.array([0.0, 0.0, -1.0]), inverse=True).astype(np.double)
    return (q, dq, quat, v, omega, gvec)


def pd_control(
    target_q: np.ndarray,
    q: np.ndarray,
    kp: np.ndarray,
    target_dq: np.ndarray,
    dq: np.ndarray,
    kd: np.ndarray,
    default: np.ndarray,
) -> np.ndarray:
    """Calculates torques from position commands.

    Args:
        target_q: The target joint positions.
        q: The current joint positions.
        kp: The position gain.
        target_dq: The target joint velocities.
        dq: The current joint velocities.
        kd: The velocity gain.
        default: The default joint positions.

    Returns:
        The calculated torques.
    """
    return kp * (target_q + default - q) - kd * dq


def run_mujoco(cfg: Any, joint_id: int = 0, steps: int = 1000) -> None:  # noqa: ANN401
    """Run the Mujoco motor identification.

    Args:
        cfg: The configuration object containing simulation settings.
        joint_id: The joint id to calibrate.
        steps: The number of steps to run the simulation.

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
    
    f = open('obs.csv', 'w')
    w = writer(f, delimiter=',')
    w.writerow(['target_pos_real', 'target_pos_sim', 'pos_sim'])

    try:
        data.qpos = model.keyframe("default").qpos
        default = deepcopy(model.keyframe("default").qpos)[-cfg.num_actions :]
        print("Default position:", default)
    except Exception as _:
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
        q, dq, _, _, _, _ = get_obs(data)
        q = q[-cfg.num_actions :]
        dq = dq[-cfg.num_actions :]

        # sin_pos = torch.tensor(math.sin(2 * math.pi * step * dt / cfg.sim_config.cycle_time))
        sin_pos = (torch.tensor(math.sin(2 * math.pi * (step + 1880) * dt / cfg.sim_config.cycle_time)) + 1) / 2

        # N hz control loop
        if step % cfg.sim_config.decimation == 0:
            target_q[joint_id] = sin_pos
            w.writerow([int(target_q[joint_id] * 1024 + 1024), target_q[joint_id], q[joint_id]])
            print(int(target_q[joint_id] * 1024 + 1024), ',', target_q[joint_id], q[joint_id])

        time_data.append(step)
        position_data.append(q[joint_id])

        # Generate PD control
        tau = pd_control(target_q, q, cfg.robot_config.kps, default, dq, cfg.robot_config.kds, default)  # Calc torques

        tau = np.clip(tau, -cfg.robot_config.tau_limit, cfg.robot_config.tau_limit)  # Clamp torques

        data.ctrl = tau

        mujoco.mj_step(model, data)
        step += 1
        viewer.render()

    f.close()
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

        class env:  # noqa: N801
            num_actions = len(robot.all_joints())
            frame_stack = 15
            c_frame_stack = 3
            num_single_obs = 11 + num_actions * c_frame_stack
            num_observations = int(frame_stack * num_single_obs)

        class sim_config:  # noqa: N801
            sim_duration = 60.0
            dt = 0.001
            decimation = 20
            cycle_time = 2.5

        class robot_config:  # noqa: N801
            tau_factor = 0.85
            tau_limit = np.array(list(robot.stiffness().values()) + list(robot.stiffness().values())) * tau_factor
            kps = tau_limit
            kds = np.array(list(robot.damping().values()) + list(robot.damping().values()))

        class normalization:  # noqa: N801
            class obs_scales:  # noqa: N801
                lin_vel = 2.0
                ang_vel = 1.0
                dof_pos = 1.0
                dof_vel = 0.05

            clip_observations = 18.0
            clip_actions = 18.0

        class control:  # noqa: N801
            action_scale = 0.25

    run_mujoco(Sim2simCfg(), joint_id=0, steps=1000*20)
