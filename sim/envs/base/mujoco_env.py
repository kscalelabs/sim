

import mujoco
import mujoco_viewer
import numpy as np
import onnxruntime as ort
from dataclasses import dataclass
from typing import Dict, Tuple
from scipy.spatial.transform import Rotation as R

from sim.envs.base.base_env import Env
from sim.envs.base.legged_robot_config import LeggedRobotCfg
from sim.env_helpers import debug_robot_state
from sim.envs.humanoids.stompymicro_config import StompyMicroCfg
from sim.utils.cmd_manager import CommandManager

np.set_printoptions(precision=2, suppress=True)


@dataclass
class MujocoCfg(StompyMicroCfg):  # LeggedRobotCfg):
    class gains:
        tau_limits: np.ndarray = np.ones((16,))
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


def simulate_env(env: MujocoEnv, policy: ort.InferenceSession, cfg: MujocoCfg, cmd_manager: CommandManager) -> None:
    """
    Run a policy in the Mujoco environment.
    
    Args:
        env: MujocoEnv instance
        policy: ONNX policy for controlled simulation
        cfg: Simulation configuration
    """
    obs = env.reset()
    count = 0
    fall_count = 0
    
    # Initialize policy state
    target_q = np.zeros(env.num_joints)
    prev_actions = np.zeros(env.num_joints)
    hist_obs = np.zeros(policy.get_metadata()["num_observations"])
    
    commands = cmd_manager.update(cfg.sim.dt)
    
    while count * cfg.sim.dt < cfg.env.episode_length_s:
        q, dq, quat, v, omega, euler = obs
        phase = count * cfg.sim.dt / cfg.rewards.cycle_time
        command_input = np.array(
            [np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase), 0, 0, 0]
        )
        obs_buf = np.concatenate([command_input, q, dq, prev_actions, omega, euler])
        policy_output = policy({
            "x_vel.1": np.array([command_input[2]], dtype=np.float32),
            "y_vel.1": np.array([command_input[3]], dtype=np.float32),
            "rot.1": np.array([command_input[4]], dtype=np.float32),
            "t.1": np.array([count * cfg.sim.dt], dtype=np.float32),
            "dof_pos.1": (q - env.default_joint_pos).astype(np.float32),
            "dof_vel.1": dq.astype(np.float32),
            "prev_actions.1": prev_actions.astype(np.float32),
            "imu_ang_vel.1": omega.astype(np.float32),
            "imu_euler_xyz.1": euler.astype(np.float32),
            "buffer.1": hist_obs.astype(np.float32),
        })
        
        target_q = policy_output["actions_scaled"]
        prev_actions = policy_output["actions"]
        hist_obs = policy_output["x.3"]

        commands = cmd_manager.update(cfg.sim.dt)
        
        obs, _, done, info = env.step(target_q)
        count += 1
        
        if info.get('fall', False):
            print("Robot fell, resetting...")
            fall_count += 1
            obs = env.reset()
            target_q = np.zeros(env.num_joints)
            prev_actions = np.zeros(env.num_joints)
            hist_obs = np.zeros(policy.get_metadata()["num_observations"])
        
        if count % 17 == 0:
            print(obs_buf)
            debug_robot_state(obs_buf, policy_output["actions"])
    
    print(f"\nSimulation complete:")
    print(f"Duration: {count * cfg.sim.dt:.1f}s")
    print(f"Falls: {fall_count}")


if __name__ == "__main__":
    from dataclasses import dataclass
    from typing import Dict, Tuple

    import mujoco
    import mujoco_viewer
    import numpy as np
    from kinfer.export.pytorch import export_to_onnx
    from kinfer.inference.python import ONNXModel
    from scipy.spatial.transform import Rotation as R

    from sim.model_export import ActorCfg
    from sim.sim2sim.helpers import get_actor_policy

    cfg = MujocoCfg()
    cfg.env.num_envs = 1
    env = MujocoEnv(cfg, render=True)
    cmd_manager = CommandManager(num_envs=cfg.env.num_envs, mode="fixed", default_cmd=[0.0, 0.0, 0.0, 0.0])
    LOAD_MODEL_PATH = "policy_1.pt" 

    policy_cfg = ActorCfg(embodiment=cfg.asset.name)

    actor_model, sim2sim_info, input_tensors = get_actor_policy(LOAD_MODEL_PATH, policy_cfg)
    export_config = {**vars(policy_cfg), **sim2sim_info}
    print(export_config)

    export_to_onnx(actor_model, input_tensors=input_tensors, config=export_config, save_path="kinfer_test.onnx")
    policy = ONNXModel("kinfer_test.onnx")

    simulate_env(env, policy, cfg, cmd_manager)
