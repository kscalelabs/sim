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
python sim/humanoid_gym/play.py --task only_legs_ppo --sim_device cpu
python sim/humanoid_gym/sims2sim.py --load_model policy_1.pt
"""
import math
from collections import deque
from copy import deepcopy

import mujoco
import mujoco_viewer
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

from sim.humanoid_gym.envs import OnlyLegsCfg
from sim.stompy_legs.joints import Stompy

JOINT_NAMES = [
    "left hip pitch",
    "left hip yaw",
    "left hip roll",
    "left knee pitch",
    "left ankle pitch",
    "left ankle roll",
    "right hip pitch",
    "right hip yaw",
    "right hip roll",
    "right knee pitch",
    "right ankle pitch",
    "right ankle roll",
]


class cmd:
    vx = -0.0
    vy = -0.3
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


def run_mujoco(policy, cfg):
    """
    Run the Mujoco simulation using the provided policy and configuration.

    Args:
        policy: The policy used for controlling the simulation.
        cfg: The configuration object containing simulation settings.

    Returns:
        None
    """
    model = mujoco.MjModel.from_xml_path(cfg.sim_config.mujoco_model_path)
    model.opt.timestep = cfg.sim_config.dt
    data = mujoco.MjData(model)

    data.qpos = model.keyframe("default").qpos
    default = deepcopy(model.keyframe("default").qpos)[-cfg.env.num_actions :]
    mujoco.mj_step(model, data)

    data.qvel = np.zeros_like(data.qvel)
    data.qacc = np.zeros_like(data.qacc)
    viewer = mujoco_viewer.MujocoViewer(model, data)

    for ii in JOINT_NAMES:
        print(data.joint(ii).id, data.joint(ii).qpos)

    target_q = np.zeros((cfg.env.num_actions), dtype=np.double)
    action = np.zeros((cfg.env.num_actions), dtype=np.double)

    hist_obs = deque()
    for _ in range(cfg.env.frame_stack):
        hist_obs.append(np.zeros([1, cfg.env.num_single_obs], dtype=np.double))

    count_lowlevel = 0

    for _ in tqdm(range(int(cfg.sim_config.sim_duration / cfg.sim_config.dt)), desc="Simulating..."):

        # Obtain an observation
        q, dq, quat, v, omega, gvec = get_obs(data)
        q = q[-cfg.env.num_actions :]
        dq = dq[-cfg.env.num_actions :]

        # 1000hz -> 100hz
        if count_lowlevel % cfg.sim_config.decimation == 0:

            obs = np.zeros([1, cfg.env.num_single_obs], dtype=np.float32)
            eu_ang = quaternion_to_euler_array(quat)
            eu_ang[eu_ang > math.pi] -= 2 * math.pi

            obs[0, 0] = math.sin(2 * math.pi * count_lowlevel * cfg.sim_config.dt / 0.64)
            obs[0, 1] = math.cos(2 * math.pi * count_lowlevel * cfg.sim_config.dt / 0.64)
            obs[0, 2] = cmd.vx * cfg.normalization.obs_scales.lin_vel
            obs[0, 3] = cmd.vy * cfg.normalization.obs_scales.lin_vel
            obs[0, 4] = cmd.dyaw * cfg.normalization.obs_scales.ang_vel
            # pfb30 - missing bit
            obs[0, 5:17] = (q - default) * cfg.normalization.obs_scales.dof_pos
            obs[0, 17:29] = dq * cfg.normalization.obs_scales.dof_vel
            obs[0, 29:41] = action
            obs[0, 41:44] = omega
            obs[0, 44:47] = eu_ang

            obs = np.clip(obs, -cfg.normalization.clip_observations, cfg.normalization.clip_observations)

            hist_obs.append(obs)
            hist_obs.popleft()

            policy_input = np.zeros([1, cfg.env.num_observations], dtype=np.float32)
            for i in range(cfg.env.frame_stack):
                policy_input[0, i * cfg.env.num_single_obs : (i + 1) * cfg.env.num_single_obs] = hist_obs[i][0, :]
            action[:] = policy(torch.tensor(policy_input))[0].detach().numpy()

            action = np.clip(action, -cfg.normalization.clip_actions, cfg.normalization.clip_actions)

            target_q = action * cfg.control.action_scale

        target_dq = np.zeros((cfg.env.num_actions), dtype=np.double)

        # Generate PD control
        tau = pd_control(
            target_q, q, cfg.robot_config.kps, target_dq, dq, cfg.robot_config.kds, default
        )  # Calc torques

        tau = np.clip(tau, -cfg.robot_config.tau_limit, cfg.robot_config.tau_limit)  # Clamp torques

        data.ctrl = tau

        mujoco.mj_step(model, data)
        viewer.render()
        count_lowlevel += 1

    viewer.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Deployment script.")
    parser.add_argument("--load_model", type=str, required=True, help="Run to load from.")
    parser.add_argument("--terrain", action="store_true", help="terrain or plane")
    parser.add_argument("--load_actions", action="store_true", help="saved_actions")
    args = parser.parse_args()

    class Sim2simCfg(OnlyLegsCfg):

        class sim_config:
            mujoco_model_path = f"sim/stompy_legs/robot_fixed.xml"
            sim_duration = 60.0
            dt = 0.001
            decimation = 10

        # pfb30 - todo this should be update more often
        class robot_config:
            tau_factor = 0.85
            tau_limit = np.array(list(Stompy.stiffness().values()) + list(Stompy.stiffness().values())) * tau_factor
            kps = tau_limit
            kds = np.array(list(Stompy.damping().values()) + list(Stompy.damping().values()))

    policy = torch.jit.load(args.load_model)
    run_mujoco(policy, Sim2simCfg())
