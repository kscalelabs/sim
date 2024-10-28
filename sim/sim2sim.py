# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2024 Beijing RobotEra TECHNOLOGY CO.,LTD. All rights reserved.

"""
Difference setup
python sim/play.py --task mini_ppo --sim_device cpu
python sim/sim2sim.py --load_model examples/standing_pro.pt --embodiment stompypro
python sim/sim2sim.py --load_model examples/standing_micro.pt --embodiment stompymicro
"""
import argparse
import math
import os
from collections import deque
from copy import deepcopy
from typing import Any, Tuple

import mujoco
import mujoco_viewer
import numpy as np
import pygame
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
import onnxruntime as ort
from sim.scripts.create_mjcf import load_embodiment
from model_export import ActorCfg, convert

import torch  # isort: skip
import matplotlib.pyplot as plt  # Add this import


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


# Wesley - this should be handled by model_export
class Sim2simCfg:
    def __init__(
        self,
        embodiment: str,
        frame_stack: int = 15,
        c_frame_stack: int = 3,
        sim_duration: float = 60.0,
        dt: float = 0.001,
        decimation: int = 10,
        cycle_time: float = 0.4,
        tau_factor: float = 3,
        lin_vel: float = 2.0,
        ang_vel: float = 1.0,
        dof_pos: float = 1.0,
        dof_vel: float = 0.05,
        clip_observations: float = 18.0,
        clip_actions: float = 18.0,
        action_scale: float = 0.25,
    ) -> None:
        self.robot = load_embodiment(embodiment)

        self.num_actions = len(self.robot.all_joints())

        self.frame_stack = frame_stack
        self.c_frame_stack = c_frame_stack
        self.num_single_obs = 11 + self.num_actions * self.c_frame_stack
        self.num_observations = int(self.frame_stack * self.num_single_obs)

        self.sim_duration = sim_duration
        self.dt = dt
        self.decimation = decimation

        self.cycle_time = cycle_time

        self.tau_factor = tau_factor
        self.tau_limit = (
            np.array(list(self.robot.effort().values()) + list(self.robot.effort().values())) * self.tau_factor
        )
        self.kps = np.array(list(self.robot.stiffness().values()) + list(self.robot.stiffness().values()))
        self.kds = np.array(list(self.robot.damping().values()) + list(self.robot.damping().values()))

        self.lin_vel = lin_vel
        self.ang_vel = ang_vel
        self.dof_pos = dof_pos
        self.dof_vel = dof_vel

        self.clip_observations = clip_observations
        self.clip_actions = clip_actions

        self.action_scale = action_scale


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


def pd_control(target_q: np.ndarray, q: np.ndarray, kp: np.ndarray, target_dq: np.ndarray, dq: np.ndarray, kd: np.ndarray, default: np.ndarray) -> np.ndarray:
    """Calculates torques from position commands"""
    return kp * (target_q + default - q) - kd * dq

def get_policy_output(policy: Any, input_data: Any) -> np.ndarray:
    if isinstance(policy, torch.jit._script.RecursiveScriptModule):
        return policy(torch.tensor(input_data))[0].detach().numpy()
    else:
        ort_inputs = {policy.get_inputs()[0].name: input_data}
        return policy.run(None, ort_inputs)[0][0]

def run_mujoco(policy: Any, cfg: Sim2simCfg, keyboard_use: bool = False) -> None:
    """
    Run the Mujoco simulation using the provided policy and configuration.

    Args:
        policy: The policy used for controlling the simulation.
        cfg: The configuration object containing simulation settings.

    Returns:
        None
    """
    model_dir = os.environ.get("MODEL_DIR", "sim/resources")
    mujoco_model_path = f"{model_dir}/{args.embodiment}/robot_fixed.xml"

    model = mujoco.MjModel.from_xml_path(mujoco_model_path)
    model.opt.timestep = cfg.dt
    data = mujoco.MjData(model)

    try:
        data.qpos = model.keyframe("default").qpos
        default = deepcopy(model.keyframe("default").qpos)[-cfg.num_actions :]
        print("Default position:", default)
    except:
        print("No default position found, using zero initialization")
        default = np.zeros(cfg.num_actions)  # 3 for pos, 4 for quat, cfg.num_actions for joints

    mujoco.mj_step(model, data)
    for ii in range(len(data.ctrl) + 1):
        print(data.joint(ii).id, data.joint(ii).name)

    data.qvel = np.zeros_like(data.qvel)
    data.qacc = np.zeros_like(data.qacc)
    viewer = mujoco_viewer.MujocoViewer(model, data)

    target_q = np.zeros((cfg.num_actions), dtype=np.double)
    action = np.zeros((cfg.num_actions), dtype=np.double)

    count_lowlevel = 0

    input_data = {
        "x_vel.1": np.zeros(1).astype(np.float32),
        "y_vel.1": np.zeros(1).astype(np.float32),
        "rot.1": np.zeros(1).astype(np.float32),
        "t.1": np.zeros(1).astype(np.float32),
        "dof_pos.1": np.zeros(cfg.num_actions).astype(np.float32),
        "dof_vel.1": np.zeros(cfg.num_actions).astype(np.float32),
        "prev_actions.1": np.zeros(cfg.num_actions).astype(np.float32),
        "imu_ang_vel.1": np.zeros(3).astype(np.float32),
        "imu_euler_xyz.1": np.zeros(3).astype(np.float32),
        "buffer.1": np.zeros(cfg.num_observations).astype(np.float32),
    }       

    positions_list = []  # List to store positions

    for _  in tqdm(range(int(cfg.sim_duration / cfg.dt)), desc="Simulating..."):
        if keyboard_use:
            handle_keyboard_input()

        # Obtain an observation
        q, dq, quat, v, omega, gvec = get_obs(data)
        q = q[-cfg.num_actions :]
        dq = dq[-cfg.num_actions :]

        # hist_obs = deque()
        # for _ in range(cfg.frame_stack):
            # hist_obs.append(np.zeros([1, cfg.num_single_obs], dtype=np.double))

        # 1000hz -> 50hz
        if count_lowlevel % cfg.decimation == 0:
            # obs = np.zeros([1, cfg.num_single_obs], dtype=np.float32)
            # eu_ang = quaternion_to_euler_array(quat)
            # eu_ang[eu_ang > math.pi] -= 2 * math.pi

            # cur_pos_obs = (q - default) * cfg.dof_pos

            # cur_vel_obs = dq * cfg.dof_vel

            # obs[0, 0] = math.sin(2 * math.pi * count_lowlevel * cfg.dt / cfg.cycle_time)
            # obs[0, 1] = math.cos(2 * math.pi * count_lowlevel * cfg.dt / cfg.cycle_time)
            # obs[0, 2] = x_vel_cmd * cfg.lin_vel
            # obs[0, 3] = y_vel_cmd * cfg.lin_vel
            # obs[0, 4] = yaw_vel_cmd * cfg.ang_vel
            # obs[0, 5 : (cfg.num_actions + 5)] = cur_pos_obs
            # obs[0, (cfg.num_actions + 5) : (2 * cfg.num_actions + 5)] = cur_vel_obs
            # obs[0, (2 * cfg.num_actions + 5) : (3 * cfg.num_actions + 5)] = action
            # obs[0, (3 * cfg.num_actions + 5) : (3 * cfg.num_actions + 5) + 3] = omega
            # obs[0, (3 * cfg.num_actions + 5) + 3 : (3 * cfg.num_actions + 5) + 2 * 3] = eu_ang

            # obs = np.clip(obs, -cfg.clip_observations, cfg.clip_observations)

            # hist_obs.append(obs)
            # hist_obs.popleft()

            # policy_input = np.zeros([1, cfg.num_observations], dtype=np.float32)
            # for i in range(cfg.frame_stack):
            #     policy_input[0, i * cfg.num_single_obs : (i + 1) * cfg.num_single_obs] = hist_obs[i][0, :]

            # action[:] = get_policy_output(policy, policy_input)
            # action = np.clip(action, -cfg.clip_actions, cfg.clip_actions)
            # target_q = action * cfg.action_scale
            positions, actions, hist_obs, input = policy.run(None, input_data)
            positions_list.append(positions)  # Store positions

            eu_ang = quaternion_to_euler_array(quat)
            eu_ang[eu_ang > math.pi] -= 2 * math.pi

            cur_pos_obs = (q - default) * cfg.dof_pos

            cur_vel_obs = dq * cfg.dof_vel

            input_data["x_vel.1"] = np.array([x_vel_cmd], dtype=np.float32)
            input_data["y_vel.1"] = np.array([y_vel_cmd], dtype=np.float32)
            input_data["rot.1"] = np.array([yaw_vel_cmd], dtype=np.float32)

            input_data["t.1"] = np.array([count_lowlevel * cfg.dt], dtype=np.float32)

            input_data["dof_pos.1"] = cur_pos_obs.astype(np.float32)
            input_data["dof_vel.1"] = cur_vel_obs.astype(np.float32)

            input_data["prev_actions.1"] = actions.astype(np.float32)

            input_data["imu_ang_vel.1"] = omega.astype(np.float32)
            input_data["imu_euler_xyz.1"] = eu_ang.astype(np.float32)

            input_data["buffer.1"] = hist_obs.astype(np.float32)

            target_q = positions
        target_dq = np.zeros((cfg.num_actions), dtype=np.double)

        # Generate PD control
        tau = pd_control(target_q, q, cfg.kps, target_dq, dq, cfg.kds, default)  # Calc torques

        tau = np.clip(tau, -cfg.tau_limit, cfg.tau_limit)  # Clamp torques
        # print(tau)
        # print(eu_ang)
        # print(x_vel_cmd, y_vel_cmd, yaw_vel_cmd)

        data.ctrl = tau

        mujoco.mj_step(model, data)
        viewer.render()
        count_lowlevel += 1

    viewer.close()

    # Plot the positions
    plt.figure(figsize=(10, 6))
    plt.plot(positions_list)
    plt.title("Joint Positions Over Time")
    plt.xlabel("Time Steps")
    plt.ylabel("Joint Positions")
    plt.legend([f"Joint {i}" for i in range(cfg.num_actions)])
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deployment script.")
    parser.add_argument("--load_model", type=str, required=True, help="Run to load from.")
    parser.add_argument("--embodiment", type=str, required=True, help="embodiment")
    parser.add_argument("--terrain", action="store_true", help="terrain or plane")
    parser.add_argument("--load_actions", action="store_true", help="saved_actions")
    parser.add_argument("--keyboard_use", action="store_true", help="keyboard_use")
    args = parser.parse_args()
            
    if args.keyboard_use:
        x_vel_cmd, y_vel_cmd, yaw_vel_cmd = 0.0, 0.0, 0.0
        pygame.init()
        pygame.display.set_caption("Simulation Control")
    else:
        x_vel_cmd, y_vel_cmd, yaw_vel_cmd = 0.4, 0.0, 0.0

    policy_cfg = ActorCfg(embodiment=args.embodiment)

    if args.embodiment == "stompypro":
        policy_cfg.cycle_time = 0.4
        policy_cfg.policy_dt = 1
    elif args.embodiment == "stompymicro":
        policy_cfg.cycle_time = 0.2

    policy = convert(args.load_model, policy_cfg)

    # policy = torch.jit.load(args.load_model)

    if args.embodiment == "stompypro":
        cfg = Sim2simCfg(
            args.embodiment,
            sim_duration=60.0,
            dt=0.001,
            decimation=10,
            cycle_time=0.4,
            tau_factor=3.0,
        )
    elif args.embodiment == "stompymicro":
        cfg = Sim2simCfg(
            args.embodiment,
            sim_duration=60.0,
            dt=0.001,
            decimation=10,
            cycle_time=0.2,
            tau_factor=2,
        )

    run_mujoco(policy, cfg, args.keyboard_use)

