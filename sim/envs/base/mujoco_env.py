import mujoco
import mujoco_viewer
import numpy as np
import onnxruntime as ort
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
from scipy.spatial.transform import Rotation as R

from sim.envs.base.base_env import Env
from sim.envs.base.legged_robot_config import LeggedRobotCfg
from sim.envs.humanoids.stompymicro_config import StompyMicroCfg

np.set_printoptions(precision=2, suppress=True)


@dataclass
class MujocoCfg(StompyMicroCfg):  # LeggedRobotCfg):
    class gains:
        tau_factor: float = 4
        kp_scale: float = 3.0
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
        
        self.kps = np.array([stiffness[joint.split('_', 1)[1]] for joint in self.robot.all_joints()]) * self.cfg.gains.kp_scale
        self.kds = np.array([damping[joint.split('_', 1)[1]] for joint in self.robot.all_joints()]) * self.cfg.gains.kd_scale
        self.tau_limits = np.array([effort[joint.split('_', 1)[1]] for joint in self.robot.all_joints()]) * self.cfg.gains.tau_factor
        
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
        
        if self.step_count % 3 == 0:  # TODO: Remove this hack
            self.render()

        # for _ in range(self.cfg.control.decimation):
            # Apply control
        mujoco.mj_step(self.model, self.data)
        
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

    def render(self):
        if self.viewer:
            self.viewer.render()

    def _compute_reward(self, obs: Tuple[np.ndarray, ...]) -> float:
        """Compute reward based on observation
        To clarify: These are vibes-based rewards to help the parameter optimizer for sim2sim. It tries to maximize this function's output. They have not been tuned properly, but do seem roughly the right order of magnitude."""
        _, _, _, v, _, euler = obs
        
        orientation_error = np.sum(np.square(euler[:2]))
        orientation_reward = np.exp(-self.cfg.rewards.tracking_sigma * orientation_error)
        
        height_error = abs(self.data.qpos[2] - self.cfg.rewards.base_height_target)
        height_reward = np.exp(-self.cfg.rewards.tracking_sigma * height_error)
        
        # Velocity penalty
        vel_magnitude = np.sum(np.square(v))
        vel_penalty = -0.1 * np.clip(vel_magnitude / 1.0, 0, 1)
        action_norm = np.sum(np.square(self.data.ctrl))
        action_smoothness = -0.1 * np.clip(action_norm / 100.0, 0, 1)
        torque_norm = np.sum(np.square(self.data.qfrc_applied[-self.num_joints:]))
        torques = -0.1 * np.clip(torque_norm / 100.0, 0, 1)
        vel_norm = np.sum(np.square(self.data.qvel[-self.num_joints:]))
        dof_vel = -0.1 * np.clip(vel_norm / 10.0, 0, 1)
        acc_norm = np.sum(np.square(self.data.qacc[-self.num_joints:]))
        dof_acc = -0.1 * np.clip(acc_norm / 100.0, 0, 1)
        return (
            0.4 * orientation_reward +
            0.4 * height_reward +
            0.05 * vel_penalty +
            0.05 * action_smoothness +
            0.05 * torques +
            0.025 * dof_vel +
            0.025 * dof_acc
        )

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
