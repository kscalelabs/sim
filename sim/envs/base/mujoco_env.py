import mujoco
import mujoco_viewer
import numpy as np
from dataclasses import dataclass
from typing import Tuple
from scipy.spatial.transform import Rotation as R
from collections import deque

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
        
        self.default_dof_pos = np.array([
            self.robot.default_standing().get(joint, 0.0) 
            for joint in self.robot.all_joints()
        ])
        print(f"Default joint positions: {self.default_dof_pos}")

        self.num_joints = len(self.robot.all_joints())
        
        self.obs_history = deque(maxlen=self.cfg.env.frame_stack)
        self.commands = np.zeros(4)  # [vel_x, vel_y, ang_vel_yaw, heading]
        self.last_actions = np.zeros(self.num_joints)
        
        self.commands_scale = np.array([
            self.cfg.normalization.obs_scales.lin_vel,
            self.cfg.normalization.obs_scales.lin_vel, 
            self.cfg.normalization.obs_scales.ang_vel
        ])

        self.compute_observations()

        print("\n=== Mujoco Robot Properties ===")
        print("PD Gains:")
        print(f"kp_scale: {self.cfg.gains.kp_scale}")
        print(f"kd_scale: {self.cfg.gains.kd_scale}")
        print(f"Raw kp values: {self.kps}")
        print(f"Raw kd values: {self.kds}")
        print(f"Tau limits: {self.tau_limits}")

        # Print mass and inertia for each body
        print("\nBody Properties:")
        for i in range(self.model.nbody):
            name = self.model.body(i).name
            mass = self.model.body(i).mass[0]
            inertia = [
                self.model.body_inertia[i][0],  # ixx
                self.model.body_inertia[i][1],  # iyy 
                self.model.body_inertia[i][2]   # izz
            ]
            quat = self.model.body(i).quat
            pos = self.model.body(i).pos
            print(f"{name}: mass={mass:.3f} | inertia={inertia} | pos={pos} | quat={quat}")

        print("\nFriction Properties:")
        for i in range(self.model.ngeom):
            name = self.model.geom(i).name
            friction = self.model.geom(i).friction
            print(f"{name}: friction={friction}")

    def compute_observations(self) -> np.ndarray:
        """Compute observations matching StompyMicroEnv implementation"""
        phase = self.episode_length_buf * self.cfg.sim.dt / self.cfg.rewards.cycle_time
        sin_pos = np.sin(2 * np.pi * phase)
        cos_pos = np.cos(2 * np.pi * phase)
        
        quat = self.data.sensor("orientation").data[[1, 2, 3, 0]]  # wxyz to xyzw
        r = R.from_quat(quat)
        
        q = (self.data.qpos[-self.num_joints:] - self.default_dof_pos) * self.cfg.normalization.obs_scales.dof_pos
        dq = self.data.qvel[-self.num_joints:] * self.cfg.normalization.obs_scales.dof_vel
        
        # Get base velocities in base frame with same scaling
        base_ang_vel = self.data.sensor("angular-velocity").data * self.cfg.normalization.obs_scales.ang_vel
        base_euler = r.as_euler('xyz') * self.cfg.normalization.obs_scales.quat
        
        # Match Isaac's command input format
        command_input = np.array([sin_pos, cos_pos] + list(self.commands[:3] * self.commands_scale))
        
        # Match Isaac's observation concatenation order
        obs_buf = np.concatenate([
            command_input.flatten(),     # 5D
            q.flatten(),                 # num_joints D
            dq.flatten(),                # num_joints D
            self.last_actions.flatten(), # num_joints D (previous actions)
            base_ang_vel.flatten(),      # 3D
            base_euler.flatten()         # 3D
        ])
        
        # Apply same observation clipping
        obs_buf = np.clip(
            obs_buf,
            -self.cfg.normalization.clip_observations,
            self.cfg.normalization.clip_observations
        )
        
        # Update observation history
        self.obs_history.append(obs_buf)
        
        # Stack history exactly like Isaac
        # First ensure we have a full history
        while len(self.obs_history) < self.obs_history.maxlen:
            self.obs_history.append(obs_buf)
            
        # Stack all frames together like Isaac does
        obs_history_array = np.stack([self.obs_history[i] for i in range(self.obs_history.maxlen)])
        obs_buf_all = obs_history_array.reshape(-1)  # will be 59 * frame_stack dimensions
        
        return obs_buf_all[np.newaxis, :]  # Return with batch dimension
    
    def reset(self) -> Tuple[np.ndarray, ...]:
        """Reset environment matching Isaac's initialization"""
        # Reset episode counter
        self.episode_length_buf = 0
        
        try:
            self.data.qpos = self.model.keyframe("default").qpos
            self.data.qpos[-self.num_joints:] = self.default_dof_pos
        except KeyError:
            self.data.qpos = np.zeros_like(self.data.qpos)
            self.data.qpos[0:3] = np.array([0.0, 0.0, self.cfg.rewards.base_height_target])
            self.data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])  # Initial quaternion
            self.data.qpos[-self.num_joints:] = self.default_dof_pos.copy()

        # Add same position noise as Isaac
        self.data.qpos[-self.num_joints:] += np.random.uniform(
            -self.cfg.domain_rand.start_pos_noise,
            self.cfg.domain_rand.start_pos_noise,
            size=self.num_joints
        )
        
        self.data.qvel = np.zeros_like(self.data.qvel)
        self.data.qacc = np.zeros_like(self.data.qacc)
        self.data.ctrl = np.zeros_like(self.data.ctrl)
        
        # Reset action history - must be zeros for first observation
        self.actions = np.zeros(self.num_joints)
        self.last_actions = np.zeros(self.num_joints)

        # Take an initial step to get valid sensor readings
        mujoco.mj_step(self.model, self.data)
        
        # Then take a step with zero actions like Isaac
        obs, _, _, _ = self.step(np.zeros(self.num_joints))
        
        return obs

    def step(self, action: np.ndarray):
        """Execute environment step matching Isaac's implementation more closely"""
        self.episode_length_buf += 1
        
        # 1. First level of clipping
        actions = np.clip(action, -self.cfg.normalization.clip_actions, self.cfg.normalization.clip_actions)
        
        # 2. Apply action delay
        delay = np.random.rand(1) * self.cfg.domain_rand.action_delay
        actions = (1 - delay) * actions + delay * self.actions
        
        # 3. Apply noise scaled by actions (matching Isaac)
        actions += self.cfg.domain_rand.action_noise * np.random.randn(*actions.shape) * actions
        
        # 4. Store clipped actions
        self.last_actions = self.actions.copy()
        self.actions = np.clip(actions, -self.cfg.normalization.clip_actions, self.cfg.normalization.clip_actions)

        # 5. Apply action scaling before PD control
        scaled_actions = self.actions * self.cfg.control.action_scale
        
        self.data.ctrl = self._compute_torques(scaled_actions)
        mujoco.mj_step(self.model, self.data)
        
        self.render()

        # Rest of the function remains the same...
        obs_buf = self.compute_observations()
        reward = 0
        done = self._check_termination(obs_buf)
        info = {
            'step': self.episode_length_buf,
            'fall': done and self.episode_length_buf < self.cfg.env.episode_length_s / self.cfg.sim.dt
        }
        return obs_buf, reward, done, info

    def render(self):
        if self.viewer:
            self.viewer.render()

    def close(self):
        if self.viewer:
            self.viewer.close()

    ## Helpers ##

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
        # Check episode length
        if self.cfg.env.episode_length_s and self.episode_length_buf >= self.cfg.env.episode_length_s / self.cfg.sim.dt:
            return True

        # Check minimum height
        if self.data.qpos[2] < self.cfg.asset.termination_height:
            return True
        
        return False

    def _compute_torques(self, actions: np.ndarray) -> np.ndarray:
        """Calculate torques from position commands"""
        torques = self.kps * (actions + self.default_dof_pos - self.data.qpos[-self.num_joints:]) - self.kds * self.data.qvel[-self.num_joints:]
        return np.clip(torques, -self.tau_limits, self.tau_limits)
