import os
from collections import deque
from copy import deepcopy
from typing import Any, Tuple, Union

import mujoco
import mujoco_viewer
import numpy as np
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

from sim.envs.humanoids.simple_stompymicro_env import StompyMicroEnv
from sim.envs.humanoids.stompymicro_config import StompyMicroCfg
from sim.utils.args_parsing import parse_args_with_extras
from sim.utils.cmd_manager import CommandManager
from sim.env_helpers import debug_robot_state

from isaacgym.torch_utils import (  # isort: skip
    get_axis_params,
    quat_apply,
    quat_conjugate,
    quat_mul,
    quat_rotate_inverse,
    to_torch,
    torch_rand_float,
)


import torch  # isort: skip


class MujocoCfg(StompyMicroCfg):
    """Configuration for sim2sim transfer"""

    def __init__(self, tau_factor: float = 3):
        super().__init__()
        self.asset.robot
        self.tau_factor = tau_factor
        self.tau_limit = (
            np.array(list(self.asset.robot.effort().values()) + list(self.asset.robot.effort().values()))
            * self.tau_factor
        )
        self.kps = np.array(list(self.asset.robot.stiffness().values()) + list(self.asset.robot.stiffness().values()))
        self.kds = np.array(list(self.asset.robot.damping().values()) + list(self.asset.robot.damping().values()))

        self.robot_joint_order = self.asset.robot.all_joints()
        self.mujoco_joint_order = None  # Will be set after MuJoCo model is loaded

        MUJOCO_JOINT_INDICES = {
            # Right arm (1-3 in mujoco)
            "right_shoulder_pitch": 0,  # 1-1
            "right_shoulder_yaw": 1,  # 2-1
            "right_elbow_yaw": 2,  # 3-1
            # Left arm (4-6 in mujoco)
            "left_shoulder_pitch": 3,  # 4-1
            "left_shoulder_yaw": 4,  # 5-1
            "left_elbow_yaw": 5,  # 6-1
            # Right leg (7-11 in mujoco)
            "right_hip_pitch": 6,  # 7-1
            "right_hip_yaw": 7,  # 8-1
            "right_hip_roll": 8,  # 9-1
            "right_knee_pitch": 9,  # 10-1
            "right_ankle_pitch": 10,  # 11-1
            # Left leg (12-16 in mujoco)
            "left_hip_pitch": 11,  # 12-1
            "left_hip_yaw": 12,  # 13-1
            "left_hip_roll": 13,  # 14-1
            "left_knee_pitch": 14,  # 15-1
            "left_ankle_pitch": 15,  # 16-1
        }

        ISAAC_JOINT_INDICES = {
            # Left leg (0-4 in isaac)
            "left_hip_pitch": 0,
            "left_hip_yaw": 1,
            "left_hip_roll": 2,
            "left_knee_pitch": 3,
            "left_ankle_pitch": 4,
            # Left arm (5-7 in isaac)
            "left_shoulder_pitch": 5,
            "left_shoulder_yaw": 6,
            "left_elbow_yaw": 7,
            # Right leg (8-12 in isaac)
            "right_hip_pitch": 8,
            "right_hip_yaw": 9,
            "right_hip_roll": 10,
            "right_knee_pitch": 11,
            "right_ankle_pitch": 12,
            # Right arm (13-15 in isaac)
            "right_shoulder_pitch": 13,
            "right_shoulder_yaw": 14,
            "right_elbow_yaw": 15,
        }

        self.mujoco_to_isaac_indices = np.zeros(self.env.num_actions, dtype=np.int32)
        self.isaac_to_mujoco_indices = np.zeros(self.env.num_actions, dtype=np.int32)

        for joint_name in MUJOCO_JOINT_INDICES.keys():
            mujoco_idx = MUJOCO_JOINT_INDICES[joint_name]
            isaac_idx = ISAAC_JOINT_INDICES[joint_name]
            self.mujoco_to_isaac_indices[mujoco_idx] = isaac_idx
            self.isaac_to_mujoco_indices[isaac_idx] = mujoco_idx

        print("Mujoco to Isaac indices:", self.mujoco_to_isaac_indices)
        print("Isaac to Mujoco indices:", self.isaac_to_mujoco_indices)


class MujocoEnv(StompyMicroEnv):
    """Handles all Mujoco-specific simulation logic"""

    def __init__(
        self,
        cfg: MujocoCfg,
        model_path: str,
        sim_params=None,
        physics_engine=None,
        sim_device: str = "cpu",
        headless: bool = False,
    ):
        self.cfg = cfg
        self.device = sim_device

        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.model.opt.timestep = cfg.sim.dt
        self.data = mujoco.MjData(self.model)
        self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)

        # Store MuJoCo joint order in config
        self.cfg.mujoco_joint_order = [self.model.actuator(i).name for i in range(self.model.nu)]
        print(
            f"MuJoCo joint order: {self.cfg.mujoco_joint_order}"
        )  # ['right_shoulder_pitch', 'right_shoulder_yaw', 'right_elbow_yaw', 'left_shoulder_pitch', 'left_shoulder_yaw', 'left_elbow_yaw', 'right_hip_pitch', 'right_hip_yaw', 'right_hip_roll', 'right_knee_pitch', 'right_ankle_pitch', 'left_hip_pitch', 'left_hip_yaw', 'left_hip_roll', 'left_knee_pitch', 'left_ankle_pitch']

        self.dof_pos = np.zeros(cfg.env.num_actions, dtype=np.float32)
        self.dof_vel = np.zeros(cfg.env.num_actions, dtype=np.float32)
        self.actions = np.zeros(cfg.env.num_actions, dtype=np.float32)
        self.default_dof_pos = np.zeros(cfg.env.num_actions, dtype=np.float32)
        default_pos_dict = cfg.asset.robot.default_standing()
        for i, joint_name in enumerate(cfg.asset.robot.all_joints()):
            isaac_idx = cfg.mujoco_to_isaac_indices[i]
            self.default_dof_pos[isaac_idx] = default_pos_dict[joint_name]
        self.ref_dof_pos = np.zeros(cfg.env.num_actions, dtype=np.float32)
        self.episode_length = 0

        mujoco_order = self.cfg.mujoco_joint_order
        self.legs_joints = {}
        self.arms_joints = {}
        for name, joint in cfg.asset.robot.legs.left.joints_motors():
            joint_idx = mujoco_order.index(joint)
            self.legs_joints["left_" + name] = joint_idx
        for name, joint in cfg.asset.robot.legs.right.joints_motors():
            joint_idx = mujoco_order.index(joint)
            self.legs_joints["right_" + name] = joint_idx

        self._initialize_state()

        base_init_state_list = (
            self.cfg.init_state.pos
            + self.cfg.init_state.rot
            + self.cfg.init_state.lin_vel
            + self.cfg.init_state.ang_vel
        )
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)

        ## Initialize Buffers ##

        self.episode_length_buf = torch.zeros(self.cfg.env.num_envs, device=self.device, dtype=torch.long)
        self.commands = torch.zeros(
            self.cfg.env.num_envs,
            self.cfg.commands.num_commands,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )  # x vel, y vel, yaw vel, heading
        self.commands_scale = torch.tensor(
            [
                self.cfg.normalization.obs_scales.lin_vel,
                self.cfg.normalization.obs_scales.lin_vel,
                self.cfg.normalization.obs_scales.ang_vel,
            ],
            device=self.device,
            requires_grad=False,
        )  # TODO change this

        self.base_quat = torch.Tensor(self.cfg.asset.robot.rotation).to(self.device)

        env_ids = torch.tensor(range(self.cfg.env.num_envs), device=self.device)

        self.root_states = torch.zeros(self.cfg.env.num_envs, 13, device=self.device, dtype=torch.float)
        self.env_origins = torch.zeros(self.cfg.env.num_envs, 3, device=self.device, requires_grad=False)

        self.root_states[env_ids] = self.base_init_state
        self.root_states[env_ids, :3] += self.env_origins[env_ids]

        self.imu_indices = None  # TODO: FIX
        if self.imu_indices:
            self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.rigid_state[:, self.imu_indices, 7:10])
            self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.rigid_state[:, self.imu_indices, 10:13])
        else:
            self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
            self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])

    def _initialize_state(self):
        """Initialize simulation state"""
        try:
            self.data.qpos = self.model.keyframe("default").qpos
            self.default_qpos = deepcopy(self.model.keyframe("default").qpos)
        except:
            print("No default position found, using zero initialization")
            self.default_qpos = np.zeros_like(self.data.qpos)

        print("Default position:", self.default_qpos[-self.cfg.env.num_actions :])
        if self.default_qpos is not None:
            print("Loaded joint positions:")
            for i, joint_name in enumerate(self.cfg.asset.robot.all_joints()):
                print(f"{joint_name}: {self.default_qpos[-len(self.cfg.asset.robot.all_joints()) + i]}")

        self.data.qvel = np.zeros_like(self.data.qvel)
        self.data.qacc = np.zeros_like(self.data.qacc)
        mujoco.mj_step(self.model, self.data)

        # Print joint information
        for ii in range(len(self.data.ctrl) + 1):
            print(self.data.joint(ii).id, self.data.joint(ii).name)

    def get_state(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Extract current simulation state in Isaac order"""
        quat = self.data.sensor("orientation").data[[1, 2, 3, 0]]
        r = R.from_quat(quat)
        base_lin_vel = r.apply(self.data.qvel[:3], inverse=True)
        base_ang_vel = self.data.sensor("angular-velocity").data
        gravity_vec = r.apply(np.array([0.0, 0.0, -1.0]), inverse=True)

        # Get raw state in Mujoco order
        mujoco_dof_pos = self.data.qpos[-self.cfg.env.num_actions :]
        mujoco_dof_vel = self.data.qvel[-self.cfg.env.num_actions :]

        # Convert to Isaac order
        self.dof_pos = mujoco_dof_pos[self.cfg.mujoco_to_isaac_indices]
        self.dof_vel = mujoco_dof_vel[self.cfg.mujoco_to_isaac_indices]

        dof_pos = self.dof_pos
        dof_vel = self.dof_vel
        quat = quat.astype(np.double)
        base_lin_vel = base_lin_vel.astype(np.double)
        base_ang_vel = base_ang_vel.astype(np.double)
        gravity_vec = gravity_vec.astype(np.double)

        return dof_pos, dof_vel, quat, base_lin_vel, base_ang_vel, gravity_vec

    def compute_observations(self):
        phase = self._get_phase()
        self.compute_ref_state()

        sin_pos = torch.sin(2 * torch.pi * phase).unsqueeze(1)
        cos_pos = torch.cos(2 * torch.pi * phase).unsqueeze(1)

        self.command_input = torch.cat((sin_pos, cos_pos, self.commands[:, :3] * self.commands_scale), dim=1)
        q = (self.dof_pos - self.default_dof_pos) * self.cfg.normalization.obs_scales.dof_pos
        dq = self.dof_vel * self.cfg.normalization.obs_scales.dof_vel

        obs_buf = torch.cat(
            (
                self.command_input,  # 5 = 2D(sin cos) + 3D(vel_x, vel_y, aug_vel_yaw)
                q,  # 16D
                dq,  # 16D
                self.actions,  # 16D
                self.base_ang_vel * self.cfg.normalization.obs_scales.ang_vel,  # 3
                self.base_euler_xyz * self.cfg.normalization.obs_scales.quat,  # 3
            ),
            dim=-1,
        )  # 59D

        if self.add_noise:
            obs_now = obs_buf.clone() + torch.randn_like(obs_buf) * self.noise_scale_vec * self.cfg.noise.noise_level
        else:
            obs_now = obs_buf.clone()
        self.obs_history.append(obs_now)

        obs_buf_all = torch.stack([self.obs_history[i] for i in range(self.obs_history.maxlen)], dim=1)  # N,T,K

        self.obs_buf = obs_buf_all.reshape(self.cfg.env.num_envs, -1)  # N, T*K

    def compute_ref_state(self):
        phase = self._get_phase()
        sin_pos = torch.sin(2 * torch.pi * phase)
        sin_pos_l = sin_pos.clone()
        sin_pos_r = sin_pos.clone()

        print(f"{self.default_dof_pos = }")
        print(f"{self.cfg.env.num_envs = }")

        self.ref_dof_pos = torch.from_numpy(self.default_dof_pos).unsqueeze(0).repeat(self.cfg.env.num_envs, 1)

        scale_1 = self.cfg.rewards.target_joint_pos_scale
        scale_2 = 2 * scale_1
        # left foot stance phase set to default joint pos
        sin_pos_l[sin_pos_l > 0] = 0
        self.ref_dof_pos[:, self.legs_joints["left_hip_pitch"]] += sin_pos_l * scale_1
        self.ref_dof_pos[:, self.legs_joints["left_knee_pitch"]] += sin_pos_l * scale_2
        self.ref_dof_pos[:, self.legs_joints["left_ankle_pitch"]] += sin_pos_l * scale_1

        # right foot stance phase set to default joint pos
        sin_pos_r[sin_pos_r < 0] = 0
        self.ref_dof_pos[:, self.legs_joints["right_hip_pitch"]] += sin_pos_r * scale_1
        self.ref_dof_pos[:, self.legs_joints["right_knee_pitch"]] += sin_pos_r * scale_2
        self.ref_dof_pos[:, self.legs_joints["right_ankle_pitch"]] += sin_pos_r * scale_1

        # Double support phase
        self.ref_dof_pos[torch.abs(sin_pos) < 0.1] = 0

        self.ref_action = 2 * self.ref_dof_pos

    def step(self, actions: torch.Tensor):
        """Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        # Reference action addition if enabled
        if self.cfg.env.use_ref_actions:
            actions += self.ref_action

        # Initial action clipping
        actions = torch.clip(actions, -self.cfg.normalization.clip_actions, self.cfg.normalization.clip_actions)

        # Final action clipping and storage
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)

        # Convert actions from Isaac to Mujoco order
        actions_mujoco = self.actions[:, self.cfg.isaac_to_mujoco_indices]

        # Render if viewer exists
        if hasattr(self, "viewer"):
            self.viewer.render()

        # Step physics with decimation
        for _ in range(self.cfg.control.decimation):
            # Compute torques (maintaining your existing PD control)
            torques = self._pd_control(
                actions_mujoco * self.cfg.action_scale,
                self.dof_pos[:, self.cfg.isaac_to_mujoco_indices],
                self.cfg.kps,
                torch.zeros_like(actions_mujoco),
                self.dof_vel[:, self.cfg.isaac_to_mujoco_indices],
                self.cfg.kds,
                self.cfg.asset.robot.default_standing(),
            )

            # Clip torques
            clipped_torques = torch.clip(torques, -self.cfg.tau_limit, self.cfg.tau_limit)

            # Apply torques and step simulation
            self.data.ctrl = clipped_torques.cpu().numpy()
            mujoco.mj_step(self.model, self.data)

            # Update any state tensors if needed
            self._update_state_tensors()

        # Post-physics computations
        self.post_physics_step()

        # Clip observations
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)

        # Update episode length
        self.episode_length += 1

        # Return observations, privileged observations, rewards, resets, and extras
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def post_physics_step(self):
        """Compute observations
        calls self._post_physics_step_callback() for common computations
        """
        # self.gym.refresh_actor_root_state_tensor(self.sim)
        # self.gym.refresh_net_contact_force_tensor(self.sim)
        # self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        # TODO(pfb30) - debug this
        origin = torch.tensor(self.cfg.init_state.rot, device=self.device).repeat(self.num_envs, 1)
        origin = quat_conjugate(origin)

        if self.imu_indices:
            self.base_quat = quat_mul(origin, self.rigid_state[:, self.imu_indices, 3:7])
            self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.rigid_state[:, self.imu_indices, 7:10])
            self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.rigid_state[:, self.imu_indices, 10:13])
        else:
            self.base_quat = self.root_states[:, 3:7]
            self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
            self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])

        self.base_euler_xyz = get_euler_xyz_tensor(self.base_quat)
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        self._post_physics_step_callback()

        self.compute_observations()

        self.last_last_actions[:] = torch.clone(self.last_actions[:])
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]
        self.last_rigid_state[:] = self.rigid_state[:]

    def _update_state_tensors(self):
        """Update any state tensors after physics step - implement based on your needs"""
        # Example - implement based on what state you need to track:
        # self.dof_pos = torch.from_numpy(self.data.qpos).to(self.device)
        # self.dof_vel = torch.from_numpy(self.data.qvel).to(self.device)
        pass

    def close(self):
        """Clean up resources"""
        self.viewer.close()

    def _get_phase(self):
        cycle_time = self.cfg.rewards.cycle_time
        phase = self.episode_length_buf * self.cfg.sim.dt / cycle_time
        return phase

    def get_contact_forces(self):
        contact_force = np.zeros(2)
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            for foot_idx, foot_name in enumerate(["foot_left", "foot_right"]):
                if (
                    self.model.geom(contact.geom1).name is not None and foot_name in self.model.geom(contact.geom1).name
                ) or (
                    self.model.geom(contact.geom2).name is not None and foot_name in self.model.geom(contact.geom2).name
                ):
                    force = np.zeros(6)
                    mujoco.mj_contactForce(self.model, self.data, i, force)
                    contact_force[foot_idx] = force[2]  # vertical force
        return contact_force > 50

    @staticmethod
    def _quat_to_euler(quat: np.ndarray) -> np.ndarray:
        """Convert quaternion to euler angles"""
        x, y, z, w = quat

        # Roll (x-axis rotation)
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2.0 * (w * y - z * x)
        pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))

        # Yaw (z-axis rotation)
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return np.array([roll, pitch, yaw])

    def _pd_control(self, target_q, q, kp, target_dq, dq, kd, default_dict):
        """PD control calculation"""
        # Convert default dictionary to array in correct order
        default = np.array([default_dict[joint_name] for joint_name in self.cfg.asset.robot.all_joints()])
        return kp * (target_q + default - q) + kd * (target_dq - dq)


class Policy:
    """Handles policy loading, observation stacking, and inference"""

    def __init__(self, cfg: MujocoCfg, model_path: str):
        self.cfg = cfg
        self.model_path = model_path
        self.policy = self._load_policy()

        if self.cfg.env.num_single_obs is not None:
            self.obs_history = deque(maxlen=self.cfg.env.frame_stack)
            # Initialize observation history
            for _ in range(self.cfg.env.frame_stack):
                self.obs_history.append(np.zeros([1, self.cfg.env.num_single_obs], dtype=np.float32))

    def _load_policy(self) -> Union[torch.jit._script.RecursiveScriptModule, Any]:
        if "pt" in self.model_path:
            return torch.jit.load(self.model_path)
        elif "onnx" in self.model_path:
            import onnxruntime as ort

            return ort.InferenceSession(self.model_path)
        else:
            raise ValueError(f"Unsupported model type: {self.model_path}")

    def __call__(self, obs: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """Compute action from observation"""
        # Convert torch tensor to numpy if needed
        if isinstance(obs, torch.Tensor):
            obs = obs.numpy()

        # If frame stacking is enabled
        if hasattr(self, "obs_history"):
            # Update observation history
            self.obs_history.append(obs.reshape(1, -1))

            # Stack observations
            policy_input = np.zeros([1, self.num_single_obs * self.cfg.env.frame_stack], dtype=np.float32)
            for i in range(self.cfg.env.frame_stack):
                policy_input[0, i * self.num_single_obs : (i + 1) * self.num_single_obs] = self.obs_history[i][0, :]
        else:
            # Use observation directly if no frame stacking
            policy_input = obs

        # Run inference based on model type
        if isinstance(self.policy, torch.jit._script.RecursiveScriptModule):
            return self.policy(torch.tensor(policy_input))[0].detach().numpy()
        else:
            ort_inputs = {self.policy.get_inputs()[0].name: policy_input}
            return self.policy.run(None, ort_inputs)[0][0]


def run_simulation(cfg: MujocoCfg, policy_path: str, command_mode: str = "fixed", legs_only: bool = False):
    """Main simulation loop"""
    # Initialize components
    model_dir = os.environ.get("MODEL_DIR") or "sim/resources"
    env = MujocoEnv(cfg, model_path=f"{model_dir}/{cfg.asset.name}/robot" + ("_fixed" if legs_only else "") + ".xml")
    policy = Policy(cfg, policy_path)
    cmd_manager = CommandManager(num_envs=cfg.env.num_envs, mode=command_mode)

    obs = env.compute_observations()
    for t in tqdm(range(int(cfg.env.episode_length_s / cfg.sim.dt)), desc="Simulating..."):
        actions = policy(obs.detach())
        commands = cmd_manager.update(cfg.sim.dt)
        env.commands[:] = commands
        obs, critic_obs, rews, dones, infos = env.step(actions.detach())
        if t % 143 == 0:
            debug_robot_state(obs)

    env.close()
    cmd_manager.close()


def add_sim2sim_arguments(parser):
    """Add sim2sim-specific arguments."""
    parser.add_argument("--load_model", type=str, required=True, help="Path to model file")
    parser.add_argument(
        "--command_mode",
        type=str,
        default="fixed",
        choices=["fixed", "oscillating", "keyboard", "random"],
        help="Command mode for robot control",
    )


if __name__ == "__main__":
    args = parse_args_with_extras(add_sim2sim_arguments)
    print("Arguments:", vars(args))
    cfg = MujocoCfg(tau_factor=10.0)
    run_simulation(cfg, args.load_model, args.command_mode)
