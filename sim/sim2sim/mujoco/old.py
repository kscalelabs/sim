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
from kinfer.export.pytorch import export_to_onnx
from kinfer.inference.python import ONNXModel
from sim.model_export import ActorCfg
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
from sim.sim2sim.helpers import get_actor_policy


@dataclass
class Sim2simCfg:
    sim_duration: float = 60.0
    dt: float = 0.001
    decimation: int = 10
    tau_factor: float = 3
    cycle_time: float = 0.25


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


def get_obs(data: mujoco.MjData) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extracts an observation from the mujoco data structure"""
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
    dq: np.ndarray,
    kd: np.ndarray,
    default: np.ndarray,
) -> np.ndarray:
    """Calculates torques from position commands"""
    return kp * (target_q + default - q) - kd * dq


def run_mujoco(
    embodiment: str,
    policy: ort.InferenceSession,
    cfg: Sim2simCfg,
    model_info: Dict[str, Union[float, List[float], str]],
) -> None:
    """
    Run the Mujoco simulation using the provided policy and configuration.
    """
    model_dir = os.environ.get("MODEL_DIR", "sim/resources")
    mujoco_model_path = f"{model_dir}/{embodiment}/robot.xml"

    model = mujoco.MjModel.from_xml_path(mujoco_model_path)
    model.opt.timestep = cfg.dt
    data = mujoco.MjData(model)

    assert isinstance(model_info["num_actions"], int)
    assert isinstance(model_info["num_observations"], int)
    assert isinstance(model_info["robot_effort"], list)
    assert isinstance(model_info["robot_stiffness"], list)
    assert isinstance(model_info["robot_damping"], list)

    tau_limit = np.array(list(model_info["robot_effort"]) + list(model_info["robot_effort"])) * cfg.tau_factor
    kps = np.array(list(model_info["robot_stiffness"]) + list(model_info["robot_stiffness"]))
    kds = np.array(list(model_info["robot_damping"]) + list(model_info["robot_damping"]))

    try:
        data.qpos = model.keyframe("default").qpos
        default = deepcopy(model.keyframe("default").qpos)[-model_info["num_actions"] :]
        print("Default position:", default)
    except:
        print("No default position found, using zero initialization")
        default = np.zeros(model_info["num_actions"])  # 3 for pos, 4 for quat, cfg.num_actions for joints
    default += np.random.uniform(-0.03, 0.03, size=default.shape)
    print("Default position:", default)
    mujoco.mj_step(model, data)
    for ii in range(len(data.ctrl) + 1):
        print(data.joint(ii).id, data.joint(ii).name)

    data.qvel = np.zeros_like(data.qvel)
    data.qacc = np.zeros_like(data.qacc)

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
        "dof_pos.1": np.zeros(model_info["num_actions"]).astype(np.float32),
        "dof_vel.1": np.zeros(model_info["num_actions"]).astype(np.float32),
        "prev_actions.1": np.zeros(model_info["num_actions"]).astype(np.float32),
        "imu_ang_vel.1": np.zeros(3).astype(np.float32),
        "imu_euler_xyz.1": np.zeros(3).astype(np.float32),
        "buffer.1": np.zeros(model_info["num_observations"]).astype(np.float32),
    }

    # Initialize variables for tracking upright steps and average speed
    upright_steps = 0
    total_speed = 0.0
    step_count = 0

    for _ in tqdm(range(int(cfg.sim_duration / cfg.dt)), desc="Simulating..."):
        # Obtain an observation
        q, dq, quat, v, omega, gvec = get_obs(data)
        q = q[-model_info["num_actions"] :]
        dq = dq[-model_info["num_actions"] :]

        eu_ang = quaternion_to_euler_array(quat)
        eu_ang[eu_ang > math.pi] -= 2 * math.pi

        # Check if the robot is upright (roll and pitch within ±30 degrees)
        if abs(eu_ang[0]) > math.radians(30) or abs(eu_ang[1]) > math.radians(30):
            print("Robot tilted heavily, ending simulation.")
            break
        else:
            upright_steps += 1  # Increment upright steps

        # Calculate speed and accumulate for average speed calculation
        speed = np.linalg.norm(v[:2])  # Speed in the x-y plane
        total_speed += speed
        step_count += 1

        # 1000hz -> 50hz
        if count_lowlevel % cfg.decimation == 0:
            # Convert sim coordinates to policy coordinates
            cur_pos_obs = q - default
            cur_vel_obs = dq

            input_data["x_vel.1"] = np.array([x_vel_cmd], dtype=np.float32)
            input_data["y_vel.1"] = np.array([y_vel_cmd], dtype=np.float32)
            input_data["rot.1"] = np.array([yaw_vel_cmd], dtype=np.float32)

            input_data["t.1"] = np.array([count_lowlevel * cfg.dt], dtype=np.float32)

            input_data["dof_pos.1"] = cur_pos_obs.astype(np.float32)
            input_data["dof_vel.1"] = cur_vel_obs.astype(np.float32)

            input_data["prev_actions.1"] = prev_actions.astype(np.float32)

            input_data["imu_ang_vel.1"] = omega.astype(np.float32)
            input_data["imu_euler_xyz.1"] = eu_ang.astype(np.float32)

            input_data["buffer.1"] = hist_obs.astype(np.float32)

            policy_output = policy(input_data)
            positions = policy_output["actions_scaled"]
            curr_actions = policy_output["actions"]
            hist_obs = policy_output["x.3"]

            target_q = positions

            prev_actions = curr_actions

        # Generate PD control
        tau = pd_control(target_q, q, kps, dq, kds, default)  # Calc torques
        tau = np.clip(tau, -tau_limit, tau_limit)  # Clamp torques

        data.ctrl = tau
        mujoco.mj_step(model, data)

        viewer.render()
        count_lowlevel += 1

    viewer.close()

    # Calculate average speed
    if step_count > 0:
        average_speed = total_speed / step_count
    else:
        average_speed = 0.0

    # Save or print the statistics at the end of the episode
    print(f"Number of upright steps: {upright_steps}")
    print(f"Average speed: {average_speed:.4f} m/s")


if __name__ == "__main__":
    x_vel_cmd, y_vel_cmd, yaw_vel_cmd = 0.2, 0.0, 0.0

    policy_cfg = ActorCfg(embodiment="stompymicro")
    policy_cfg.cycle_time = 0.2
    cfg = Sim2simCfg(
        sim_duration=10.0,
        dt=0.001,
        decimation=10,
        tau_factor=2,
        cycle_time=policy_cfg.cycle_time,
    )

    actor_model, sim2sim_info, input_tensors = get_actor_policy("policy_1.pt", policy_cfg)

    # Merge policy_cfg and sim2sim_info into a single config object
    export_config = {**vars(policy_cfg), **sim2sim_info}
    print(export_config)
    export_to_onnx(actor_model, input_tensors=input_tensors, config=export_config, save_path="kinfer_test.onnx")
    policy = ONNXModel("kinfer_test.onnx")

    metadata = policy.get_metadata()

    model_info = {
        "num_actions": metadata["num_actions"],
        "num_observations": metadata["num_observations"],
        "robot_effort": metadata["robot_effort"],
        "robot_stiffness": metadata["robot_stiffness"],
        "robot_damping": metadata["robot_damping"],
    }

    run_mujoco(
        "stompymicro",
        policy,
        cfg,
        model_info,
    )