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
from kinfer.export.pytorch import export_to_onnx
from kinfer.inference.python import ONNXModel
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

from sim.env_helpers import debug_robot_state
from sim.model_export import ActorCfg, Actor
import torch  # noqa: E402

np.set_printoptions(precision=4, suppress=True)


def get_actor_policy(model_path: str, cfg: ActorCfg) -> Tuple[torch.nn.Module, dict, Tuple[torch.Tensor, ...]]:
    model = torch.jit.load(model_path, map_location="cpu")
    a_model = Actor(model, cfg)

    # Gets the model input tensors.
    x_vel = torch.randn(1)
    y_vel = torch.randn(1)
    rot = torch.randn(1)
    t = torch.randn(1)
    dof_pos = torch.randn(a_model.num_actions)
    dof_vel = torch.randn(a_model.num_actions)
    prev_actions = torch.randn(a_model.num_actions)
    imu_ang_vel = torch.randn(3)
    imu_euler_xyz = torch.randn(3)
    buffer = a_model.get_init_buffer()
    input_tensors = (x_vel, y_vel, rot, t, dof_pos, dof_vel, prev_actions, imu_ang_vel, imu_euler_xyz, buffer)

    # Add sim2sim metadata
    robot_effort = list(a_model.robot.effort().values())
    robot_stiffness = list(a_model.robot.stiffness().values())
    robot_damping = list(a_model.robot.damping().values())
    num_actions = a_model.num_actions
    num_observations = a_model.num_observations

    return (
        a_model,
        {
            "robot_effort": robot_effort,
            "robot_stiffness": robot_stiffness,
            "robot_damping": robot_damping,
            "num_actions": num_actions,
            "num_observations": num_observations,
        },
        input_tensors,
    )


@dataclass
class Sim2simCfg:
    sim_duration: float = 60.0
    dt: float = 0.001
    decimation: int = 10
    tau_factor: float = 3
    cycle_time: float = 0.25


def quaternion_to_euler(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion [x,y,z,w] to euler angles [roll,pitch,yaw]"""
    x, y, z, w = quat
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))  # (x-axis)
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))  # (y-axis)
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))  # (z-axis)
    return np.array([roll, pitch, yaw])


def get_obs(data: mujoco.MjData, num_actions: int) -> Tuple[np.ndarray, ...]:
    """Extract observation from mujoco data"""
    q = data.qpos[-num_actions:].astype(np.double)
    dq = data.qvel[-num_actions:].astype(np.double)
    quat = data.sensor("orientation").data[[1, 2, 3, 0]].astype(np.double)
    r = R.from_quat(quat)
    v = r.apply(data.qvel[:3], inverse=True).astype(np.double)
    omega = data.sensor("angular-velocity").data.astype(np.double)
    gvec = r.apply(np.array([0.0, 0.0, -1.0]), inverse=True).astype(np.double)
    return q, dq, quat, v, omega, gvec


def prepare_policy_input(
    command_input: np.ndarray,
    q: np.ndarray,
    dq: np.ndarray,
    prev_actions: np.ndarray,
    omega: np.ndarray,
    euler: np.ndarray,
    default: np.ndarray,
    obs_scales: Dict[str, float],
) -> np.ndarray:
    """Prepare observation buffer for policy input"""
    q_scaled = (q - default) * obs_scales["dof_pos"]
    dq_scaled = dq * obs_scales["dof_vel"]
    ang_vel_scaled = omega * obs_scales["ang_vel"]
    euler_scaled = euler * obs_scales["quat"]

    return np.concatenate(
        [
            command_input,  # 5D  (sin, cos, vel_x, vel_y, vel_yaw)
            q_scaled,  # 16D (joint positions)
            dq_scaled,  # 16D (joint velocities)
            prev_actions,  # 16D (previous actions)
            ang_vel_scaled,  # 3D  (base angular velocity)
            euler_scaled,  # 3D  (base euler angles)
        ]
    )


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


def reset_robot_state(model: mujoco.MjModel, data: mujoco.MjData, default: np.ndarray, num_actions: int) -> None:
    """Reset the robot to its initial state with small random perturbations."""
    try:
        data.qpos = model.keyframe("default").qpos
        # Add small random perturbation to default joint positions
        data.qpos[-num_actions:] = default + np.random.uniform(-0.03, 0.03, size=default.shape)
    except:
        # If no default keyframe, use zero initialization with small noise
        data.qpos[-num_actions:] = np.random.uniform(-0.03, 0.03, size=num_actions)
    
    # Reset velocities and accelerations
    data.qvel = np.zeros_like(data.qvel)
    data.qacc = np.zeros_like(data.qacc)
    mujoco.mj_step(model, data)


def run_mujoco(
    task: str,
    policy: ort.InferenceSession,
    cfg: Sim2simCfg,
    model_info: Dict[str, Union[float, List[float], str]],
) -> None:
    """
    Run the Mujoco simulation using the provided policy and configuration.
    """
    model_dir = os.environ.get("MODEL_DIR", "sim/resources")
    mujoco_model_path = f"{model_dir}/{task}/robot.xml"

    model = mujoco.MjModel.from_xml_path(mujoco_model_path)
    model.opt.timestep = cfg.dt
    data = mujoco.MjData(model)

    tau_limit = np.array(model_info["robot_effort"] * 2) * cfg.tau_factor
    kps = np.array(model_info["robot_stiffness"] * 2)
    kds = np.array(model_info["robot_damping"] * 2)
    print(f"Tau limit: {tau_limit}")
    print(f"Stiffness: {kps}")
    print(f"Damping: {kds}")
    
    tau_score = 300
    kps_score = 30
    kds_score = 2.5
    tau_limit = np.ones((16,)) * tau_score
    kps = np.ones((16,)) * kps_score
    kds = np.ones((16,)) * kds_score
    
    print(f"Tau limit: {tau_limit}")
    print(f"Stiffness: {kps}")
    print(f"Damping: {kds}")

    # Initialize simulation state
    try:
        data.qpos = model.keyframe("default").qpos
        default = deepcopy(model.keyframe("default").qpos)[-model_info["num_actions"] :]
        print("Initial qpos quaternion:", data.qpos[3:7])
        print("Default position:", default)
    except:
        print("No default position found, using zero initialization")
        default = np.zeros(model_info["num_actions"])  # 3 for pos, 4 for quat, cfg.num_actions for joints
    # default += np.random.uniform(-0.03, 0.03, size=default.shape)
    data.qvel = np.zeros_like(data.qvel)
    data.qacc = np.zeros_like(data.qacc)
    mujoco.mj_step(model, data)

    # Initialize viewer
    viewer = mujoco_viewer.MujocoViewer(model, data)

    # Initialize control variables
    target_q = np.zeros(model_info["num_actions"], dtype=np.double)
    prev_actions = np.zeros(model_info["num_actions"], dtype=np.double)
    hist_obs = np.zeros(model_info["num_observations"], dtype=np.double)

    # Initialize tracking variables
    total_speed = 0.0
    step_count = 0
    count_lowlevel = 0
    fall_count = 0
    max_angle_threshold = math.radians(30)

    # Observation scaling factors
    obs_scales = {"lin_vel": 2.0, "ang_vel": 1.0, "dof_pos": 1.0, "dof_vel": 0.05, "quat": 1.0}

    for t in tqdm(range(int(cfg.sim_duration / cfg.dt)), desc="Simulating..."):
        # Get current state
        q, dq, quat, v, omega, gvec = get_obs(data, model_info["num_actions"])
        euler = quaternion_to_euler(quat)
        euler[euler > math.pi] -= 2 * math.pi

        if abs(euler[0]) > max_angle_threshold or abs(euler[1]) > max_angle_threshold:
            print(f"\nRobot fell! Resetting... (Fall count: {fall_count + 1})")
            reset_robot_state(model, data, default, model_info["num_actions"])
            fall_count += 1
            
            # Reset control variables
            target_q = np.zeros(model_info["num_actions"], dtype=np.double)
            prev_actions = np.zeros(model_info["num_actions"], dtype=np.double)
            hist_obs = np.zeros(model_info["num_observations"], dtype=np.double)
            count_lowlevel = 0
            phase = 0
            continue

        # Update statistics
        speed = np.linalg.norm(v[:2])
        total_speed += speed
        step_count += 1
        
        if count_lowlevel % cfg.decimation == 0:
            phase = count_lowlevel * cfg.dt / cfg.cycle_time
            command_input = np.array(
                [np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase), x_vel_cmd, y_vel_cmd, yaw_vel_cmd]
            )

            obs_buf = prepare_policy_input(command_input, q, dq, prev_actions, omega, euler, default, obs_scales)

            # Get policy action
            policy_output = policy(
                {
                    "x_vel.1": np.array([x_vel_cmd], dtype=np.float32),
                    "y_vel.1": np.array([y_vel_cmd], dtype=np.float32),
                    "rot.1": np.array([yaw_vel_cmd], dtype=np.float32),
                    "t.1": np.array([count_lowlevel * cfg.dt], dtype=np.float32),
                    "dof_pos.1": (q - default).astype(np.float32),
                    "dof_vel.1": dq.astype(np.float32),
                    "prev_actions.1": prev_actions.astype(np.float32),
                    "imu_ang_vel.1": omega.astype(np.float32),
                    "imu_euler_xyz.1": euler.astype(np.float32),
                    "buffer.1": hist_obs.astype(np.float32),
                }
            )

            target_q = policy_output["actions_scaled"]
            prev_actions = policy_output["actions"]
            hist_obs = policy_output["x.3"]

        # Apply control
        tau = pd_control(target_q, q, kps, dq, kds, default)
        tau = np.clip(tau, -tau_limit, tau_limit)
        data.ctrl = tau

        if t % 17 == 0:
            debug_robot_state(obs_buf, tau)
        
        if t == 200:
            # raise Exception
            pass

        mujoco.mj_step(model, data)
        if count_lowlevel % 3 == 0:
            viewer.render()
        count_lowlevel += 1

    viewer.close()

    if step_count > 0:
        print(f"\nSimulation Statistics:")
        print(f"Number of falls: {fall_count}")
        print(f"Average speed: {total_speed / step_count:.4f} m/s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deployment script.")
    parser.add_argument("--task", type=str, required=True, help="Task name.")
    parser.add_argument("--load_model", type=str, required=True, help="Path to run to load from.")
    args = parser.parse_args()

    x_vel_cmd, y_vel_cmd, yaw_vel_cmd = 0.0, 0.0, 0.0

    policy_cfg = ActorCfg(embodiment=args.task)
    cfg = Sim2simCfg(
        sim_duration=10.0,
        dt=0.001,
        decimation=10,
        tau_factor=4.0,
    )
    if args.task == "gpr":
        policy_cfg.cycle_time = cfg.cycle_time = 0.25
    elif args.task == "stompymicro":
        policy_cfg.cycle_time = cfg.cycle_time = 0.5
        cfg.decimation = 20.0
        cfg.tau_factor = 4.0
    else:
        raise ValueError(f"Unknown task: {args.task}")

    actor_model, sim2sim_info, input_tensors = get_actor_policy(args.load_model, policy_cfg)
    export_config = {**vars(policy_cfg), **sim2sim_info}
    print(export_config)

    export_to_onnx(actor_model, input_tensors=input_tensors, config=export_config, save_path="kinfer_test.onnx")
    policy = ONNXModel("kinfer_test.onnx")

    metadata = policy.get_metadata()

    run_mujoco(args.task, policy, cfg, metadata)
