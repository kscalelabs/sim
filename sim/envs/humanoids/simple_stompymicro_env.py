from sim.envs.base.simple_legged_robot import LeggedRobot
from sim.resources.stompymicro.joints import Robot
from sim.envs.humanoids.stompymicro_config import StompyMicroCfg

from isaacgym.torch_utils import *  # isort: skip

import torch  # isort:skip


class StompyMicroEnv(LeggedRobot):
    """StompyFreeEnv is a class that represents a custom environment for a legged robot.

    Args:
        cfg: Configuration object for the legged robot.
        sim_params: Parameters for the simulation.
        physics_engine: Physics engin e used in the simulation.
        sim_device: Device used for the simulation.
        headless: Flag indicating whether the simulation should be run in headless mode.

    Attributes:
        last_feet_z (float): The z-coordinate of the last feet position.
        feet_height (torch.Tensor): Tensor representing the height of the feet.
        sim (gymtorch.GymSim): The simulation object.
        terrain (HumanoidTerrain): The terrain object.
        up_axis_idx (int): The index representing the up axis.
        command_input (torch.Tensor): Tensor representing the command input.
        obs_buf (torch.Tensor): Tensor representing the observations buffer.
        obs_history (collections.deque): Deque containing the history of observations.

    Methods:
        _get_phase(): Calculates the phase of the gait cycle.
        compute_ref_state(): Computes the reference state.
        create_sim(): Creates the simulation, terrain, and environments.
        step(actions): Performs a simulation step with the given actions.
        compute_observations(): Computes the observations.
        reset_idx(env_ids): Resets the environment for the specified environment IDs.
    """

    def __init__(self, cfg: StompyMicroCfg, sim_params, physics_engine, sim_device: str, headless: bool):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self.last_feet_z = self.cfg.asset.default_feet_height
        self.feet_height = torch.zeros((self.num_envs, 2), device=self.device)
        self.reset_idx(torch.tensor(range(self.num_envs), device=self.device))

        env_handle = self.envs[0]
        actor_handle = self.actor_handles[0]

        self.legs_joints = {}
        for name, joint in Robot.legs.left.joints_motors():
            joint_handle = self.gym.find_actor_dof_handle(env_handle, actor_handle, joint)
            self.legs_joints["left_" + name] = joint_handle

        for name, joint in Robot.legs.right.joints_motors():
            joint_handle = self.gym.find_actor_dof_handle(env_handle, actor_handle, joint)
            self.legs_joints["right_" + name] = joint_handle

        self.arms_joints = {}
        for name, joint in Robot.arms.left.joints_motors():
            joint_handle = self.gym.find_actor_dof_handle(env_handle, actor_handle, joint)
            self.arms_joints["left_" + name] = joint_handle

        for name, joint in Robot.arms.right.joints_motors():
            joint_handle = self.gym.find_actor_dof_handle(env_handle, actor_handle, joint)
            self.arms_joints["right_" + name] = joint_handle

        print("\nJoint Dictionary Mapping:")
        for name, handle in list(self.legs_joints.items()) + list(self.arms_joints.items()):
            print(f"{name} -> DOF {handle}")

        self.compute_observations()

    def _get_phase(self):
        cycle_time = self.cfg.rewards.cycle_time
        phase = self.episode_length_buf * self.dt / cycle_time
        return phase

    def compute_ref_state(self):
        phase = self._get_phase()
        sin_pos = torch.sin(2 * torch.pi * phase)
        sin_pos_l = sin_pos.clone()
        sin_pos_r = sin_pos.clone()
        self.ref_dof_pos = self.default_dof_pos.repeat(self.num_envs, 1)

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

    def create_sim(self):
        """Creates simulation, terrain and evironments"""
        self.up_axis_idx = 2  # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(
            self.sim_device_id,
            self.graphics_device_id,
            self.physics_engine,
            self.sim_params,
        )
        self._create_ground_plane()
        self._create_envs()

    def compute_observations(self):
        phase = self._get_phase()
        self.compute_ref_state()

        sin_pos = torch.sin(2 * torch.pi * phase).unsqueeze(1)
        cos_pos = torch.cos(2 * torch.pi * phase).unsqueeze(1)

        self.command_input = torch.cat((sin_pos, cos_pos, self.commands[:, :3] * self.commands_scale), dim=1)
        q = (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos
        dq = self.dof_vel * self.obs_scales.dof_vel

        obs_buf = torch.cat(
            (
                self.command_input,  # 5 = 2D(sin cos) + 3D(vel_x, vel_y, aug_vel_yaw)
                q,  # 16D
                dq,  # 16D
                self.actions,  # 16D
                self.base_ang_vel * self.obs_scales.ang_vel,  # 3
                self.base_euler_xyz * self.obs_scales.quat,  # 3
            ),
            dim=-1,
        )  # 59D

        obs_now = obs_buf.clone()
        self.obs_history.append(obs_now)

        obs_buf_all = torch.stack([self.obs_history[i] for i in range(self.obs_history.maxlen)], dim=1)  # N,T,K

        self.obs_buf = obs_buf_all.reshape(self.num_envs, -1)  # N, T*K
