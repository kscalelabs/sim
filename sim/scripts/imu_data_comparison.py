"""Testing the falling down IMU data comparison.

Run:
    python sim/scripts/imu_data_comparison.py --embodiment zbot2
"""
import argparse
import os
from copy import deepcopy

import matplotlib.pyplot as plt
import mediapy as media
import mujoco
import mujoco_viewer
import numpy as np
import pandas as pd
from tqdm import tqdm


def plot_comparison(sim_data: pd.DataFrame, real_data: pd.DataFrame) -> None:
    """Plot the real IMU data.

    Args:
        sim_data: The simulated IMU data.
        real_data: The real IMU data.
    """
    plt.figure(figsize=(10, 15))


    real_timestamps = (real_data['timestamp'] - real_data['timestamp'].iloc[0]).dt.total_seconds().to_numpy()
    
    # Accelerometer plots
    plt.subplot(6, 1, 1)
    plt.plot(real_timestamps, sim_data['accel_x'].to_numpy(), label='Simulated Accel X')
    plt.plot(real_timestamps, real_data['accel_x'].to_numpy(), label='Real Accel X')
    plt.title('Accelerometer X')
    plt.legend()

    plt.subplot(6, 1, 2)
    plt.plot(real_timestamps, sim_data['accel_y'].to_numpy(), label='Simulated Accel Y')
    plt.plot(real_timestamps, real_data['accel_y'].to_numpy(), label='Real Accel Y')
    plt.title('Accelerometer Y')
    plt.legend()

    plt.subplot(6, 1, 3)
    plt.plot(real_timestamps, sim_data['accel_z'].to_numpy(), label='Simulated Accel Z')
    plt.plot(real_timestamps, real_data['accel_z'].to_numpy(), label='Real Accel Z')
    plt.title('Accelerometer Z')
    plt.legend()

    # Gyroscope plots
    plt.subplot(6, 1, 4)
    plt.plot(real_timestamps, sim_data['gyro_x'].to_numpy(), label='Simulated Gyro X')
    plt.plot(real_timestamps, real_data['gyro_x'].to_numpy(), label='Real Gyro X')
    plt.title('Gyroscope X')
    plt.legend()

    plt.subplot(6, 1, 5)
    plt.plot(real_timestamps, sim_data['gyro_y'].to_numpy(), label='Simulated Gyro Y')
    plt.plot(real_timestamps, real_data['gyro_y'].to_numpy(), label='Real Gyro Y')
    plt.title('Gyroscope Y')
    plt.legend()

    plt.subplot(6, 1, 6)
    plt.plot(real_timestamps, sim_data['gyro_z'].to_numpy(), label='Simulated Gyro Z')
    plt.plot(real_timestamps, real_data['gyro_z'].to_numpy(), label='Real Gyro Z')
    plt.title('Gyroscope Z')
    plt.legend()

    plt.tight_layout()
    plt.savefig('imu_data_comparison.png')


def read_real_data(data_file: str = "sim/resources/zbot2/imu_data.csv") -> None:
    """Plot the real IMU data.

    Args:
        data_file: The path to the real IMU data file.

    Returns:
        The real IMU data.
    """
    # Reading the data from CSV file
    df = pd.read_csv(data_file)

    df = df.apply(pd.to_numeric, errors='ignore')
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    return df


def pd_control(
    target_q: np.ndarray,
    q: np.ndarray,
    kp: np.ndarray,
    dq: np.ndarray,
    kd: np.ndarray,
    default: np.ndarray,
) -> np.ndarray:
    """Calculates torques from position commands

    Args:
        target_q: The target position.
        q: The current position.
        kp: The proportional gain.
        dq: The current velocity.
        kd: The derivative gain.
        default: The default position.

    Returns:
        The calculated torques.
    """
    return kp * (target_q + default - q) - kd * dq


def run_simulation(
    embodiment: str,
    kp: float = 1.0,
    kd: float = 1.0,
    sim_duration: float = 15.0,
    effort: float = 5.0,
) -> None:
    """
    Run the Mujoco simulation using the provided policy and configuration.

    Args:
        embodiment: The embodiment to use for the simulation.
        kp: The proportional gain.
        kd: The derivative gain.
        sim_duration: The duration of the simulation.
        effort: The effort to apply to the robot.
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

    data.qpos = model.keyframe("standing").qpos
    default = deepcopy(model.keyframe("standing").qpos)[-model_info["num_actions"] :]
    print("Default position:", default)

    target_q = np.zeros((model_info["num_actions"]), dtype=np.double)
    viewer = mujoco_viewer.MujocoViewer(model, data,"offscreen")

    force_duration = 400  # Duration of force application in timesteps
    force_timer = 0

    applied_force = np.array([0.0, -3, 0.0])

    sim_data = {
        "timestamp": [],
        "gyro_x": [],
        "gyro_y": [],
        "gyro_z": [],
        "accel_x": [],
        "accel_y": [],
        "accel_z": [],
        "mag_x": [],
        "mag_y": [],
        "mag_z": [],
    }

    for timestep in tqdm(range(int(sim_duration / model_info["sim_dt"])), desc="Simulating..."):
        if timestep == 500:
            force_timer = force_duration
        if timestep % 10 == 0:
            # Keep the robot in the same position
            q = data.qpos.astype(np.double)[-model_info["num_actions"] :]
            dq = data.qvel.astype(np.double)[-model_info["num_actions"] :]
            tau = pd_control(target_q, q, kps, dq, kds, default)  # Calc torques
            tau = np.clip(tau, -tau_limit, tau_limit)  # Clamp torques
            data.ctrl = tau
            mujoco.mj_step(model, data)
            if timestep % 100 == 0:
                img = viewer.read_pixels()
                frames.append(img)

                # Obtain an observation
                gyroscope = data.sensor("angular-velocity").data.astype(np.double)
                accelerometer = data.sensor("linear-acceleration").data.astype(np.double)
                magnetometer = data.sensor("magnetometer").data.astype(np.double)

                sim_data["timestamp"].append(timestep * model_info["sim_dt"])
                sim_data["gyro_x"].append(gyroscope[0])
                sim_data["gyro_y"].append(gyroscope[1])
                sim_data["gyro_z"].append(gyroscope[2])
                sim_data["accel_x"].append(accelerometer[0])
                sim_data["accel_y"].append(accelerometer[1])
                sim_data["accel_z"].append(accelerometer[2])
                sim_data["mag_x"].append(magnetometer[0])
                sim_data["mag_y"].append(magnetometer[1])
                sim_data["mag_z"].append(magnetometer[2])

        if timestep == 12680:
            break

        if force_timer > 0:
            # Apply force if timer is active
            if force_timer > 0:
                data.xfrc_applied[1] = np.concatenate([applied_force, np.zeros(3)])
                force_timer -= 1
            else:
                data.xfrc_applied[1] = np.zeros(6)          

    media.write_video("push_tests.mp4", frames, fps=framerate)

    # sim_data["timestamp"] = np.array(sim_data["timestamp"])
    # sim_data["gyro_x"] = np.array(sim_data["gyro_x"])
    # sim_data["gyro_y"] = np.array(sim_data["gyro_y"])
    # sim_data["gyro_z"] = np.array(sim_data["gyro_z"])
    # sim_data["accel_x"] = np.array(sim_data["accel_x"])
    # sim_data["accel_y"] = np.array(sim_data["accel_y"])
    # sim_data["accel_z"] = np.array(sim_data["accel_z"])
    # sim_data["mag_x"] = np.array(sim_data["mag_x"])
    # sim_data["mag_y"] = np.array(sim_data["mag_y"])
    # sim_data["mag_z"] = np.array(sim_data["mag_z"])
    return pd.DataFrame(sim_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deployment script.")
    parser.add_argument("--embodiment", type=str, required=True, help="Embodiment name.")
    parser.add_argument("--kp", type=float, default=10.0, help="Path to run to load from.")
    parser.add_argument("--kd", type=float, default=1.0, help="Path to run to load from.")
    args = parser.parse_args()

    sim_data = run_simulation(args.embodiment, args.kp, args.kd)
    real_data = read_real_data()
    plot_comparison(sim_data, real_data)