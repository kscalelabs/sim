"""
python sim/sim2sim.py --load_model examples/standing_pro.pt --embodiment gpr
python sim/sim2sim.py --load_model examples/standing_micro.pt --embodiment zeroth
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
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

# from sim.h5_logger import HDF5Logger
from sim.model_export import ActorCfg, convert_model_to_onnx


@dataclass
class Sim2simCfg:
    sim_duration: float = 60.0
    dt: float = 0.001
    decimation: int = 10
    tau_factor: float = 3
    cycle_time: float = 0.4


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
    keyboard_use: bool = False,
    log_h5: bool = False,
    render: bool = True,
    h5_out_dir: str = "sim/resources",
) -> None:
    """
    Run the Mujoco simulation using the provided policy and configuration.

    Args:
        policy: The policy used for controlling the simulation.
        cfg: The configuration object containing simulation settings.

    Returns:
        None
    """
    model_dir = os.environ.get("MODEL_DIR", "sim/resources")
    mujoco_model_path = f"{model_dir}/{embodiment}/robot_fixed.xml"

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
        "dof_pos.1": np.zeros(model_info["num_actions"]).astype(np.float32),
        "dof_vel.1": np.zeros(model_info["num_actions"]).astype(np.float32),
        "prev_actions.1": np.zeros(model_info["num_actions"]).astype(np.float32),
        "imu_ang_vel.1": np.zeros(3).astype(np.float32),
        "imu_euler_xyz.1": np.zeros(3).astype(np.float32),
        "buffer.1": np.zeros(model_info["num_observations"]).astype(np.float32),
    }

    if log_h5:
        stop_state_log = int(cfg.sim_duration / cfg.dt) / cfg.decimation
        logger = HDF5Logger(
            data_name=embodiment,
            num_actions=model_info["num_actions"],
            max_timesteps=stop_state_log,
            num_observations=model_info["num_observations"],
            h5_out_dir=h5_out_dir
        )

    # Initialize variables for tracking upright steps and average speed
    upright_steps = 0
    total_speed = 0.0
    step_count = 0

    for _ in tqdm(range(int(cfg.sim_duration / cfg.dt)), desc="Simulating..."):
        if keyboard_use:
            handle_keyboard_input()

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

        # 1000hz -> Nhz
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

            positions, curr_actions, hist_obs = policy.run(None, input_data)
            target_q = positions
            prev_actions = curr_actions

        # Generate PD control
        tau = pd_control(target_q, q, kps, dq, kds, default)  # Calc torques
        tau = np.clip(tau, -tau_limit, tau_limit)  # Clamp torques

        data.ctrl = tau
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


def parse_modelmeta(
    modelmeta: List[Tuple[str, str]],
    verbose: bool = False,
) -> Dict[str, Union[float, List[float], str]]:
    parsed_meta: Dict[str, Union[float, List[float], str]] = {}
    for key, value in modelmeta:
        if value.startswith("[") and value.endswith("]"):
            parsed_meta[key] = list(map(float, value.strip("[]").split(",")))
        else:
            try:
                parsed_meta[key] = float(value)
                try:
                    if int(value) == parsed_meta[key]:
                        parsed_meta[key] = int(value)
                except ValueError:
                    pass
            except ValueError:
                print(f"Failed to convert {value} to float")
                parsed_meta[key] = value
    if verbose:
        for key, value in parsed_meta.items():
            print(f"{key}: {value}")
    return parsed_meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deployment script.")
    parser.add_argument("--embodiment", type=str, required=True, help="Embodiment name.")
    parser.add_argument("--load_model", type=str, required=True, help="Path to run to load from.")
    parser.add_argument("--keyboard_use", action="store_true", help="keyboard_use")
    parser.add_argument("--log_h5", action="store_true", help="log_h5")
    parser.add_argument("--h5_out_dir", type=str, default="sim/resources", help="Directory to save HDF5 files")
    parser.add_argument("--no_render", action="store_false", dest="render", help="Disable rendering")
    parser.set_defaults(render=True)
    args = parser.parse_args()

    if args.keyboard_use:
        x_vel_cmd, y_vel_cmd, yaw_vel_cmd = 0.0, 0.0, 0.0
        pygame.init()
        pygame.display.set_caption("Simulation Control")
    else:
        x_vel_cmd, y_vel_cmd, yaw_vel_cmd = 0.4, 0.0, 0.0

    policy_cfg = ActorCfg(embodiment=args.embodiment)
    if args.embodiment == "gpr":
        policy_cfg.cycle_time = 0.64
        cfg = Sim2simCfg(
            sim_duration=10.0,
            dt=0.001,
            decimation=10,
            tau_factor=4.0,
            cycle_time=policy_cfg.cycle_time,
        )
    elif args.embodiment == "zeroth":
        policy_cfg.cycle_time = 0.2
        cfg = Sim2simCfg(
            sim_duration=10.0,
            dt=0.001,
            decimation=10,
            tau_factor=1,
            cycle_time=policy_cfg.cycle_time,
        )

    if args.load_model.endswith(".onnx"):
        policy = ort.InferenceSession(args.load_model)
    else:
        policy = convert_model_to_onnx(
            args.load_model, policy_cfg, save_path="policy.onnx"
        )

    model_info = parse_modelmeta(
        policy.get_modelmeta().custom_metadata_map.items(),
        verbose=True,
    )

    run_mujoco(
        args.embodiment,
        policy,
        cfg,
        model_info,
        args.keyboard_use,
        args.log_h5,
        args.render,
        args.h5_out_dir,
    )
