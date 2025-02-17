"""Sim2sim deployment test.

Run:
    python sim/sim2sim2.py --load_model examples/gpr_walking.kinfer --embodiment gpr
    python sim/sim2sim2.py --load_model kinfer_policy.onnx --embodiment zbot2
"""

import argparse
import math
import os
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Tuple, Union

import mujoco
import mujoco_viewer
import numpy as np
import onnxruntime as ort
import pygame
import torch
from kinfer.export.pytorch import export_to_onnx
from kinfer.inference.python import ONNXModel
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm


def handle_keyboard_input() -> None:
    global x_vel_cmd, y_vel_cmd, yaw_vel_cmd

    keys = pygame.key.get_pressed()

    # Update movement commands based on arrow keys
    if keys[pygame.K_UP]:
        x_vel_cmd += 0.0005
    if keys[pygame.K_DOWN]:
        x_vel_cmd -= 0.0005
    if keys[pygame.K_LEFT]:
        y_vel_cmd += 0.0005
    if keys[pygame.K_RIGHT]:
        y_vel_cmd -= 0.0005

    # Yaw control
    if keys[pygame.K_a]:
        yaw_vel_cmd += 0.001
    if keys[pygame.K_z]:
        yaw_vel_cmd -= 0.001


def quaternion_to_euler_array(quat: np.ndarray) -> np.ndarray:
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


def get_gravity_orientation(quaternion):
    """
    Args:
        quaternion: np.ndarray[float, float, float, float]
    
    Returns:
        gravity_orientation: np.ndarray[float, float, float]
    """
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)

    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation


def get_obs(data: mujoco.MjData) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extracts an observation from the mujoco data structure"""
    q = data.qpos.astype(np.double)
    dq = data.qvel.astype(np.double)
    # quat = data.sensor("orientation").data[[1, 2, 3, 0]].astype(np.double)
    quat = data.sensor("base_link_quat").data[[1, 2, 3, 0]].astype(np.double)

    r = R.from_quat(quat)
    gvec = r.apply(np.array([0.0, 0.0, -1.0]), inverse=True).astype(np.double)
    v = r.apply(data.qvel[:3], inverse=True).astype(np.double)  # In the base frame
    # omega = data.sensor("angular-velocity").data.astype(np.double)
    omega = data.sensor("base_link_ang_vel").data.astype(np.double)

    # gvec = get_gravity_orientation(data.sensor("orientation").data)
    return (q, dq, quat, v, omega, gvec)


def pd_control(
    target_q: np.ndarray,
    target_dq: np.ndarray,
    q: np.ndarray,
    kp: np.ndarray,
    dq: np.ndarray,
    kd: np.ndarray,
    default: np.ndarray,
) -> np.ndarray:
    """Calculates torques from position commands"""
    return kp * (target_q + default - q) + kd * (target_dq - dq)


def run_mujoco(
    embodiment: str,
    policy: ort.InferenceSession,
    model_info: Dict[str, Union[float, List[float], str]],
    keyboard_use: bool = False,
    log_h5: bool = False,
    render: bool = True,
    sim_duration: float = 60.0,
    h5_out_dir: str = "sim/resources",
    terrain: bool = False,
) -> None:
    """
    Run the Mujoco simulation using the provided policy and configuration.

    Args:
        policy: The policy used for controlling the simulation.
        cfg: The configuration object containing simulation settings.
    """
    model_dir = os.environ.get("MODEL_DIR", "sim/resources")
    if terrain:
        mujoco_model_path = f"{model_dir}/{embodiment}/robot_fixed_terrain.xml"
    else:
        mujoco_model_path = f"{model_dir}/{embodiment}/robot_fixed.xml"

    model = mujoco.MjModel.from_xml_path(mujoco_model_path)
    model.opt.timestep = model_info["sim_dt"]
    data = mujoco.MjData(model)

    assert isinstance(model_info["num_actions"], int)
    assert isinstance(model_info["num_observations"], int)
    assert isinstance(model_info["robot_effort"], list)
    assert isinstance(model_info["robot_stiffness"], list)
    assert isinstance(model_info["robot_damping"], list)
    assert isinstance(model_info["num_joints"], int)

    efforts = model_info["robot_effort"]
    stiffnesses = model_info["robot_stiffness"]
    dampings = model_info["robot_damping"]
    leg_lims = [efforts[0], efforts[1], efforts[1], efforts[0], efforts[2]]
    tau_limit = np.array(leg_lims + leg_lims) * model_info["tau_factor"]

    leg_kps = [stiffnesses[0], stiffnesses[1], stiffnesses[1], stiffnesses[0], stiffnesses[2]]
    kps = np.array(leg_kps + leg_kps)

    leg_kds = [dampings[0], dampings[1], dampings[1], dampings[0], dampings[2]]
    kds = np.array(leg_kds + leg_kds)

    try:
        data.qpos = model.keyframe("default").qpos
        default = deepcopy(model.keyframe("default").qpos)[-model_info["num_joints"] :]
        print("Default position:", default)
    except:
        print("No default position found, using zero initialization")
        default = np.zeros(model_info["num_joints"])  # 3 for pos, 4 for quat, cfg.num_actions for joints
    # default += np.random.uniform(-0.03, 0.03, size=default.shape)
    print("Default position:", default)
    mujoco.mj_step(model, data)
    for ii in range(len(data.ctrl)//2 + 1):
        print(data.joint(ii).id, data.joint(ii).name)

    data.qvel = np.zeros_like(data.qvel)
    data.qacc = np.zeros_like(data.qacc)

    if render:
        viewer = mujoco_viewer.MujocoViewer(model, data)

    target_q = np.zeros((model_info["num_actions"]), dtype=np.double)
    prev_actions = np.zeros((model_info["num_actions"]), dtype=np.double)
    hist_obs = np.zeros((model_info["num_observations"]), dtype=np.double)

    count_lowlevel = 0

    input_data = {
        "x_vel.1": np.zeros(1).astype(np.float32),
        "y_vel.1": np.zeros(1).astype(np.float32),
        "rot.1": np.zeros(1).astype(np.float32),
        "t.1": np.zeros(1).astype(np.float32),
        "dof_pos.1": np.zeros(model_info["num_joints"]).astype(np.float32),
        "dof_vel.1": np.zeros(model_info["num_joints"]).astype(np.float32),
        "prev_actions.1": np.zeros(model_info["num_actions"]).astype(np.float32),
        # "imu_ang_vel.1": np.zeros(3).astype(np.float32),
        # "imu_euler_xyz.1": np.zeros(3).astype(np.float32),
        "projected_gravity.1": np.zeros(3).astype(np.float32),
        "buffer.1": np.zeros(model_info["num_observations"]).astype(np.float32),
    }

    if log_h5:
        from sim.h5_logger import HDF5Logger

        stop_state_log = int(sim_duration / model_info["sim_dt"]) / model_info["sim_decimation"]
        logger = HDF5Logger(
            data_name=embodiment,
            num_actions=model_info["num_actions"],
            max_timesteps=stop_state_log,
            num_observations=model_info["num_observations"],
            h5_out_dir=h5_out_dir,
        )

    # Initialize variables for tracking upright steps and average speed
    upright_steps = 0
    total_speed = 0.0
    step_count = 0

    for _ in tqdm(range(int(sim_duration / model_info["sim_dt"])), desc="Simulating..."):
        if keyboard_use:
            handle_keyboard_input()

        # Obtain an observation
        q, dq, quat, v, omega, gvec = get_obs(data)
        q = q[-model_info["num_joints"] :]
        dq = dq[-model_info["num_joints"] :]

        # eu_ang = quaternion_to_euler_array(quat)
        # eu_ang[eu_ang > math.pi] -= 2 * math.pi

        # eu_ang = np.array([0.0, 0.0, 0.0])
        # omega = np.array([0.0, 0.0, 0.0])
        # gvec = np.array([0.0, 0.0, -1.0])
        # Calculate speed and accumulate for average speed calculation
        speed = np.linalg.norm(v[:2])  # Speed in the x-y plane
        total_speed += speed
        step_count += 1

        # 1000hz -> 50hz
        if count_lowlevel % model_info["sim_decimation"] == 0:
            # Convert sim coordinates to policy coordinates
            cur_pos_obs = q - default
            cur_vel_obs = dq

            input_data["x_vel.1"] = np.array([x_vel_cmd], dtype=np.float32)
            input_data["y_vel.1"] = np.array([y_vel_cmd], dtype=np.float32)
            input_data["rot.1"] = np.array([yaw_vel_cmd], dtype=np.float32)

            input_data["t.1"] = np.array([count_lowlevel * model_info["sim_dt"]], dtype=np.float32)

            input_data["dof_pos.1"] = cur_pos_obs.astype(np.float32)
            input_data["dof_vel.1"] = cur_vel_obs.astype(np.float32)

            input_data["prev_actions.1"] = prev_actions.astype(np.float32)

            input_data["projected_gravity.1"] = gvec.astype(np.float32)
            # input_data["imu_ang_vel.1"] = omega.astype(np.float32)
            # input_data["imu_euler_xyz.1"] = eu_ang.astype(np.float32)

            input_data["buffer.1"] = hist_obs.astype(np.float32)

            policy_output = policy(input_data)
            positions = policy_output["actions_scaled"]
            curr_actions = policy_output["actions"]
            hist_obs = policy_output["x.3"]

            target_q = positions[:model_info["num_joints"]]
            target_dq = positions[model_info["num_joints"]:]

            target_q = (target_q + default) * 60

            prev_actions = curr_actions

        data.ctrl = np.concatenate([target_q, target_dq])
        print(data.ctrl)
        # Generate PD control
        # tau = pd_control(target_q, target_dq, q, kps, dq, kds, default)  # Calc torques
        # tau = np.clip(tau, -tau_limit, tau_limit)  # Clamp torques

        # data.ctrl = tau
        mujoco.mj_step(model, data)

        if render:
            viewer.render()
        count_lowlevel += 1

    if render:
        viewer.close()

    # Calculate average speed
    if step_count > 0:
        average_speed = total_speed / step_count
    else:
        average_speed = 0.0

    # Save or print the statistics at the end of the episode
    print(f"Number of upright steps: {upright_steps}")
    print(f"Average speed: {average_speed:.4f} m/s")

    if log_h5:
        logger.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deployment script.")
    parser.add_argument("--embodiment", type=str, required=True, help="Embodiment name.")
    parser.add_argument("--load_model", type=str, required=True, help="Path to run to load from.")
    parser.add_argument("--keyboard_use", action="store_true", help="keyboard_use")
    parser.add_argument("--log_h5", action="store_true", help="log_h5")
    parser.add_argument("--h5_out_dir", type=str, default="sim/resources", help="Directory to save HDF5 files")
    parser.add_argument("--no_render", action="store_false", dest="render", help="Disable rendering")
    parser.add_argument("--terrain", action="store_true", help="terrain")
    parser.set_defaults(render=True, terrain=False)
    
    args = parser.parse_args()

    if args.keyboard_use:
        x_vel_cmd, y_vel_cmd, yaw_vel_cmd = 0.0, 0.0, 0.0
        pygame.init()
        pygame.display.set_caption("Simulation Control")
    else:
        x_vel_cmd, y_vel_cmd, yaw_vel_cmd = 0.5, 0.0, 0.0

    policy = ONNXModel(args.load_model)
    metadata = policy.get_metadata()
    model_info = {
        "num_actions": metadata["num_actions"],
        "num_observations": metadata["num_observations"],
        "robot_effort": metadata["robot_effort"],
        "robot_stiffness": metadata["robot_stiffness"],
        "robot_damping": metadata["robot_damping"],
        "sim_dt": metadata["sim_dt"],
        "sim_decimation": metadata["sim_decimation"],
        "tau_factor": metadata["tau_factor"],
        "num_joints": metadata["num_joints"],
    }

    run_mujoco(
        embodiment=args.embodiment,
        policy=policy,
        model_info=model_info,
        keyboard_use=args.keyboard_use,
        log_h5=args.log_h5,
        render=args.render,
        h5_out_dir=args.h5_out_dir,
        terrain=args.terrain,
    )
