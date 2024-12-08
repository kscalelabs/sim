

import mujoco
import mujoco_viewer
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
from scipy.spatial.transform import Rotation as R

from sim.envs.base.base_env import Env
from sim.envs.base.legged_robot_config import LeggedRobotCfg
from sim.envs.humanoids.stompymicro_config import StompyMicroCfg


@dataclass
class MujocoCfg(StompyMicroCfg):  # LeggedRobotCfg):
    class gains:
        tau_limits: np.ndarray = np.ones((16,))
        tau_factor: float = 10
        kds: np.ndarray = np.ones((16,))
        kps: np.ndarray = np.ones((16,))
        kp_scale: float = 1.0
        kd_scale: float = 1.0


class MujocoEnv(Env):
    def __init__(
        self,
        cfg: MujocoCfg,
        render: bool = False,
    ):
        self.cfg = cfg
        self.robot = self.cfg.asset.robot
        self.model = mujoco.MjModel.from_xml_path(self.cfg.asset.file_xml)
        self.model.opt.timestep = self.cfg.sim.dt
        self.data = mujoco.MjData(self.model)
        self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data) if render else None

        stiffness = self.robot.stiffness()
        damping = self.robot.damping()
        effort = self.robot.effort()
        
        self.kps = np.array([stiffness[joint.split('_', 1)[1]] for joint in self.robot.all_joints()])
        self.kds = np.array([damping[joint.split('_', 1)[1]] for joint in self.robot.all_joints()])
        self.tau_limits = np.array([effort[joint.split('_', 1)[1]] for joint in self.robot.all_joints()])
        
        self.default_joint_pos = np.array([
            self.robot.default_standing().get(joint, 0.0) 
            for joint in self.robot.all_joints()
        ])

        self.num_joints = len(self.robot.all_joints())
        self.step_count = 0
        
    def _get_obs(self) -> Tuple[np.ndarray, ...]:
        """Extract observation from mujoco data"""
        q = self.data.qpos[-self.num_joints:].astype(np.double)
        dq = self.data.qvel[-self.num_joints:].astype(np.double)
        quat = self.data.sensor("orientation").data[[1, 2, 3, 0]].astype(np.double)
        r = R.from_quat(quat)
        v = r.apply(self.data.qvel[:3], inverse=True).astype(np.double)
        omega = self.data.sensor("angular-velocity").data.astype(np.double)
        euler = r.as_euler('xyz')
        return (q, dq, quat, v, omega, euler)

    def _pd_control(self, target_q: np.ndarray) -> np.ndarray:
        """Calculate torques from position commands"""
        q = self.data.qpos[-self.num_joints:]
        dq = self.data.qvel[-self.num_joints:]
        return self.kps * (target_q + self.default_joint_pos - q) - self.kds * dq

    def reset(self) -> Dict:
        """Reset the environment and return initial observation"""
        # Reset state
        try:
            self.data.qpos = self.model.keyframe("default").qpos
            self.data.qpos[-self.num_joints:] = self.default_joint_pos + np.random.uniform(
                -self.cfg.domain_rand.start_pos_noise,
                self.cfg.domain_rand.start_pos_noise,
                size=self.num_joints
            )
        except:
            self.data.qpos[-self.num_joints:] = np.random.uniform(
                -self.cfg.domain_rand.start_pos_noise,
                self.cfg.domain_rand.start_pos_noise,
                size=self.num_joints
            )
        
        self.data.qvel = np.zeros_like(self.data.qvel)
        self.data.qacc = np.zeros_like(self.data.qacc)
        mujoco.mj_step(self.model, self.data)
        
        self.step_count = 0
        obs = self._get_obs()
        return obs

    def step(self, action: np.ndarray) -> Tuple[Dict, Dict, float, bool, Dict]:
        """Execute one environment step"""
        self.step_count += 1
        
        # Apply action through PD control
        tau = self._pd_control(action)
        tau = np.clip(tau, -self.tau_limits, self.tau_limits)
        self.data.ctrl = tau
        
        # Step simulation
        mujoco.mj_step(self.model, self.data)
        if self.viewer and self.step_count % 3 == 0:
            self.viewer.render()
        
        # Get observation
        obs = self._get_obs()
        reward = self._compute_reward(obs)
        done = self._check_termination(obs)
        
        # Package observation and info
        obs_buf = self._get_obs()
        info = {
            'step': self.step_count,
            'fall': done and self.step_count < self.cfg.env.episode_length_s / self.cfg.sim.dt
        }
        
        return obs_buf, reward, done, info

    def _compute_reward(self, obs: Tuple[np.ndarray, ...]) -> float:
        """Compute reward based on observation"""
        _, _, _, v, _, euler = obs
        
        # Upright reward from orientation
        orientation_error = np.sum(np.square(euler[:2]))
        orientation_reward = np.exp(-self.cfg.rewards.tracking_sigma * orientation_error)
        
        # Velocity penalty
        vel_penalty = -0.1 * np.sum(np.square(v))
        
        # Height reward
        height_error = abs(self.data.qpos[2] - self.cfg.rewards.base_height_target)
        height_reward = np.exp(-self.cfg.rewards.tracking_sigma * height_error)
        
        return orientation_reward + vel_penalty + height_reward

    def _check_termination(self, obs: Tuple[np.ndarray, ...]) -> bool:
        """Check termination conditions"""
        # Check falling (when height is below cfg.asset.termination_height)
        if self.data.qpos[2] < self.cfg.asset.termination_height:
            return True

        # Check episode length
        if self.cfg.env.episode_length_s and self.step_count >= self.cfg.env.episode_length_s / self.cfg.sim.dt:
            return True

        # Check minimum height
        if self.data.qpos[2] < self.cfg.asset.termination_height:
            return True
            
        return False

    def close(self):
        if self.viewer:
            self.viewer.close()


def simulate_env(env: MujocoEnv, n_episodes: int = 3, params_to_optimize: Optional[Dict] = None):
    """Simulate environment for n_episodes"""
    obs = env.reset()
    done = False
    while not done:
        action = np.random.uniform(-1, 1, env.num_joints)
        obs, rewards, done, _ = env.step(action)
        # somehow, optimize for rewards
        print(rewards)
    env.close()
    print("Done")


if __name__ == "__main__":
    cfg = MujocoCfg()
    env = MujocoEnv(cfg, render=True)
    simulate_env(env, 3)
