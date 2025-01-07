"""Id and standing test.

Run:
    python sim/scripts/push_standing_tests.py --load_model kinfer.onnx --embodiment zbot2
"""
import argparse
import os
from copy import deepcopy

import mediapy as media
import mujoco
import mujoco_viewer
import numpy as np
from tqdm import tqdm


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


def run_test(
    embodiment: str,
    kp: float = 1.0,
    kd: float = 1.0,
    push_force: float = 1.0,
    sim_duration: float = 3.0,
    effort: float = 5.0,
) -> None:
    """
    Run the Mujoco simulation using the provided policy and configuration.

    Args:
        policy: The policy used for controlling the simulation.
        cfg: The configuration object containing simulation settings.
    """
    model_info = {
        "sim_dt": 0.001,
        "tau_factor": 2,
        "num_actions": 10,
        "num_observations": 10,
        "robot_effort": [effort] * 10,
        "robot_stiffness": [kp] * 10,
        "robot_damping": [kd] * 10,
    }
    frames = []
    framerate = 30
    model_dir = os.environ.get("MODEL_DIR", "sim/resources")
    mujoco_model_path = f"{model_dir}/{embodiment}/robot_fixed.xml"

    model = mujoco.MjModel.from_xml_path(mujoco_model_path)
    model.opt.timestep = model_info["sim_dt"]
    data = mujoco.MjData(model)

    tau_limit = np.array(list(model_info["robot_effort"])) * model_info["tau_factor"]
    kps = np.array(model_info["robot_stiffness"])
    kds = np.array(model_info["robot_damping"])
    print(kps)
    print(kds)
    print(tau_limit)

    data.qpos = model.keyframe("standing").qpos
    default = deepcopy(model.keyframe("standing").qpos)[-model_info["num_actions"] :]
    print("Default position:", default)
    
    mujoco.mj_step(model, data)
    for ii in range(len(data.ctrl) + 1):
        print(data.joint(ii).id, data.joint(ii).name)

    data.qvel = np.zeros_like(data.qvel)
    data.qacc = np.zeros_like(data.qacc)

    target_q = np.zeros((model_info["num_actions"]), dtype=np.double)
    viewer = mujoco_viewer.MujocoViewer(model, data,"offscreen")
 
    force_application_interval = 1000  # Apply force every 1000 steps (1 second at 1000Hz)
    force_magnitude_range = (-push_force, push_force) # Force range in Newtons
    force_duration = 100  # Duration of force application in timesteps
    force_timer = 0


    for timestep in tqdm(range(int(sim_duration / model_info["sim_dt"])), desc="Simulating..."):
        # Obtain an observation
        q = data.qpos.astype(np.double)[-model_info["num_actions"] :]
        dq = data.qvel.astype(np.double)[-model_info["num_actions"] :]

        # Generate PD control
        tau = pd_control(target_q, q, kps, dq, kds, default)  # Calc torques
        tau = np.clip(tau, -tau_limit, tau_limit)  # Clamp torques

        data.ctrl = tau

        # Apply random forces periodically
        if timestep % force_application_interval == 0:
            print("Applying force")
            # Generate random force vector
            random_force = np.random.uniform(
                force_magnitude_range[0], 
                force_magnitude_range[1], 
                size=3
            )
            force_timer = force_duration

        # Apply force if timer is active
        if force_timer > 0:
            data.xfrc_applied[1] = np.concatenate([random_force, np.zeros(3)])
            force_timer -= 1
        else:
            data.xfrc_applied[1] = np.zeros(6)

        mujoco.mj_step(model, data)
        if timestep % 100 == 0:
            img = viewer.read_pixels()
            frames.append(img)

        # viewer.render()
    breakpoint()
    media.write_video("push_tests.mp4", frames, fps=framerate)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deployment script.")
    parser.add_argument("--embodiment", type=str, required=True, help="Embodiment name.")
    parser.add_argument("--kp", type=float, default=17.0, help="Path to run to load from.")
    parser.add_argument("--kd", type=float, default=1.0, help="Path to run to load from.")
    parser.add_argument("--push_force", type=float, default=1.0, help="Path to run to load from.")
    args = parser.parse_args()

    run_test(args.embodiment, args.kp, args.kd, args.push_force)
