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

MUJOCO_TO_ISAAC = {
    7: 13,  # right_shoulder_pitch 
    8: 14,  # right_shoulder_yaw
    9: 15,  # right_elbow_yaw
    10: 5,  # left_shoulder_pitch
    11: 6,  # left_shoulder_yaw  
    12: 7,  # left_elbow_yaw
    13: 8,  # right_hip_pitch
    14: 9,  # right_hip_yaw
    15: 10, # right_hip_roll
    16: 11, # right_knee_pitch
    17: 12, # right_ankle_pitch
    18: 0,  # left_hip_pitch
    19: 1,  # left_hip_yaw
    20: 2,  # left_hip_roll
    21: 3,  # left_knee_pitch
    22: 4,  # left_ankle_pitch
}
ISAAC_TO_MUJOCO = {v: k for k, v in MUJOCO_TO_ISAAC.items()}


@dataclass
class MujocoCfg(StompyMicroCfg):  # LeggedRobotCfg):
    class gains:
        tau_factor: float = 4
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
        self.num_joints = len(self.robot.all_joints())
        self.model = mujoco.MjModel.from_xml_path(self.cfg.asset.file_xml)
        self.model.opt.timestep = self.cfg.sim.dt
        self.data = mujoco.MjData(self.model)
        self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data) if render else None

        self.mujoco_indices = np.array([k for k in MUJOCO_TO_ISAAC.keys()])
        self.isaac_order = np.array([MUJOCO_TO_ISAAC[k] for k in self.mujoco_indices])
        
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

        self.num_joints = len(self.robot.all_joints())
        
        ### Initialize Buffers ##
        self.obs_history = deque(maxlen=self.cfg.env.frame_stack)
        for _ in range(self.cfg.env.frame_stack):
            self.obs_history.append(np.zeros(self.cfg.env.num_single_obs))
        self.commands = np.zeros(4)  # [vel_x, vel_y, ang_vel_yaw, heading]
        self.last_actions = np.zeros(self.num_joints)
        self.origin_quat = R.from_quat(self.cfg.init_state.rot)
        self.origin_quat_inv = self.origin_quat.inv()
        ## Initialize Buffers ##
        
        self.commands_scale = np.array([
            self.cfg.normalization.obs_scales.lin_vel,
            self.cfg.normalization.obs_scales.lin_vel, 
            self.cfg.normalization.obs_scales.ang_vel
        ])

        self.reset()
        self.compute_observations()
        print_model_info(self.model, self.data)

    def compute_observations(self) -> None:
        """Compute observations matching StompyMicroEnv implementation"""
        phase = self.episode_length_buf * self.cfg.sim.dt / self.cfg.rewards.cycle_time
        sin_pos = np.sin(2 * np.pi * phase)
        cos_pos = np.cos(2 * np.pi * phase)
        command_input = np.array([sin_pos, cos_pos] + list(self.commands[:3] * self.commands_scale))
        
        q = self.data.qpos[-self.num_joints:][self.isaac_order]
        dq = self.data.qvel[-self.num_joints:][self.isaac_order]
        q = (q - self.default_dof_pos) * self.cfg.normalization.obs_scales.dof_pos
        dq = dq * self.cfg.normalization.obs_scales.dof_vel

        sensor_quat = self.data.sensor("orientation").data[[1, 2, 3, 0]]  # wxyz to xyzw
        sensor_rotation = R.from_quat(sensor_quat)

        base_rotation = self.origin_quat_inv * sensor_rotation
        base_quat = base_rotation.as_quat()  # returns xyzw quaternion
        
        ang_vel_world = self.data.sensor("angular-velocity").data
        base_ang_vel = base_rotation.inv().apply(ang_vel_world)
        base_ang_vel = base_ang_vel * self.cfg.normalization.obs_scales.ang_vel
        
        base_euler = base_rotation.as_euler('xyz') * self.cfg.normalization.obs_scales.quat
        
        assert command_input.shape == (5,)
        assert q.shape == (self.num_joints,)
        assert dq.shape == (self.num_joints,)
        assert base_ang_vel.shape == (3,)
        assert base_euler.shape == (3,)
        
        # Match Isaac's observation concatenation order
        obs_buf = np.concatenate([
            command_input.flatten(),     # 5D
            q.flatten(),                 # num_joints D
            dq.flatten(),                # num_joints D
            self.last_actions.flatten(), # num_joints D
            base_ang_vel.flatten(),      # 3D
            base_euler.flatten()         # 3D
        ])
        
        # TODO: Add noise
        
        self.obs_history.append(obs_buf)
        
        obs_buf_all = np.stack([self.obs_history[i] for i in range(self.obs_history.maxlen)], axis=1)  # Shape: (K, T)
        assert obs_buf_all.shape == (self.cfg.env.num_single_obs, self.cfg.env.frame_stack)
        
        self.obs_buf = obs_buf_all.reshape(-1)[np.newaxis, :]
        assert self.obs_buf.shape == (1, self.cfg.env.num_single_obs * self.cfg.env.frame_stack)
    
    def reset(self) -> Tuple[np.ndarray, ...]:
        """Reset environment matching Isaac's initialization"""
        # Reset episode counter
        self.episode_length_buf = 0
        self.reset_buf = False
        
        try:
            self.data.qpos = self.model.keyframe("default").qpos
            self.data.qpos[-self.num_joints:] = self.default_dof_pos
        except KeyError as e:
            print(f"Warning: Keyframe 'default' not found in model. Using zero initial state instead.")
            self.data.qpos[-self.num_joints:] = self.default_dof_pos.copy()

        self.data.qpos[-self.num_joints:] += np.random.uniform(
            -self.cfg.domain_rand.start_pos_noise,
            self.cfg.domain_rand.start_pos_noise,
            size=self.num_joints
        )
        self.data.qvel = np.zeros_like(self.data.qvel)
        self.data.qacc = np.zeros_like(self.data.qacc)
        self.data.ctrl = np.zeros_like(self.data.ctrl)
        
        self.actions = np.zeros(self.num_joints)
        self.last_actions = np.zeros(self.num_joints)

        mujoco.mj_step(self.model, self.data)
        obs, _, _, _ = self.step(np.zeros(self.num_joints))
        return obs

    def step(self, action: np.ndarray):
        """Execute environment step matching Isaac's implementation more closely"""
        actions = np.clip(action, -self.cfg.normalization.clip_actions, self.cfg.normalization.clip_actions)
        delay = np.random.rand(1) * self.cfg.domain_rand.action_delay
        actions = (1 - delay) * actions + delay * self.actions
        actions += self.cfg.domain_rand.action_noise * np.random.randn(*actions.shape) * actions
        self.actions = np.clip(action, -self.cfg.normalization.clip_actions, self.cfg.normalization.clip_actions)
        
        # self.render()
        
        # TODO: Understand why below doesn't work!
        for _ in range(self.cfg.control.decimation):
            self.render()
            self.data.ctrl = self._compute_torques(actions)
            mujoco.mj_step(self.model, self.data)
        
        ## Post Physics Step ##
        self.episode_length_buf += 1
        
        # origin and imu_indices weirdness??
        
        self.check_termination()
        self.compute_rewards()
        if self.reset_buf:
            self.reset()
        
        self.compute_observations()

        ## Post Physics Step ##

        self.obs_buf = np.clip(self.obs_buf, -self.cfg.normalization.clip_observations, self.cfg.normalization.clip_observations)

        self.extras = {
            'step': self.episode_length_buf,
            'fall': self.reset_buf and self.episode_length_buf < self.cfg.env.episode_length_s / self.cfg.sim.dt
        }
        
        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def render(self):
        if self.viewer:
            self.viewer.render()

    def close(self):
        if self.viewer:
            self.viewer.close()

    ## Helpers ##

    def compute_rewards(self) -> None:
        # TODO: Implement reward computation
        self.rew_buf = 0

    def check_termination(self) -> None:
        """Check termination conditions"""
        self.reset_buf = False
        self.reset_buf |= self.episode_length_buf >= self.cfg.env.episode_length_s / self.cfg.sim.dt
        self.reset_buf |= self.data.qpos[2] < self.cfg.asset.termination_height

    def _compute_torques(self, actions: np.ndarray) -> np.ndarray:
        """Calculate torques from position commands"""
        if actions.ndim == 2:
            actions = actions.squeeze(0)

        actions_mujoco = actions[self.isaac_order]
        
        actions_scaled = actions_mujoco * self.cfg.control.action_scale
        q_mujoco = self.data.qpos[-self.num_joints:]
        dq_mujoco = self.data.qvel[-self.num_joints:]
        
        torques = self.kps * (actions_scaled + self.default_dof_pos - q_mujoco) - self.kds * dq_mujoco
        return np.clip(torques, -self.tau_limits, self.tau_limits)


def print_model_info(model, data):
    print(f"\nMujoco DOF info:")
    print(f"qpos shape: {data.qpos.shape}")
    print(f"qvel shape: {data.qvel.shape}")
    print("\nJoint info:")
    for i in range(model.njnt):
        print(f"Joint {i}: {model.joint(i).name}, type={model.joint(i).type}, qposadr={model.joint(i).qposadr}, dofadr={model.joint(i).dofadr}")

    print("\nBody Properties:")
    for i in range(model.nbody):
        name = model.body(i).name
        mass = model.body(i).mass[0]
        inertia = [
            model.body_inertia[i][0],  # ixx
            model.body_inertia[i][1],  # iyy 
            model.body_inertia[i][2]   # izz
        ]
        quat = model.body(i).quat
        pos = model.body(i).pos
        print(f"{name}: mass={mass:.3f} | inertia={inertia} | pos={pos} | quat={quat}")

    # print("\nFriction Properties:")
    # for i in range(model.ngeom):
    #     name = model.geom(i).name
    #     friction = model.geom(i).friction
    #     print(f"{name}: friction={friction}")
