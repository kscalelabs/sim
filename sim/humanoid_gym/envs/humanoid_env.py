"""Defines the environment for training the humanoid."""

from typing import Any, NewType

import torch
from humanoid.envs import LeggedRobot
from humanoid.utils.terrain import HumanoidTerrain
from torch import Tensor

from sim.humanoid_gym.envs.humanoid_config import StompyCfg
from sim.stompy.joints import Stompy

SimParams = NewType("SimParams", Any)
SimType = NewType("SimType", Any)

NUM_JOINTS = len(Stompy.all_joints())


class StompyFreeEnv(LeggedRobot):
    """StompyFreeEnv adapts the XBot environment for training Stompy.

    Note that the underlying XBot training code has some interesting
    conventions to be aware of:

    - The function `_prepare_reward_function` looks for all the methods on this
      class that start with `_reward_` and adds them to the reward function.
      Each reward function should return a tensor of rewards.

    Parameters:
        cfg: Configuration object for the legged robot.
        sim_params: Parameters for the simulation.
        physics_engine: Physics engine used in the simulation.
        sim_device: Device used for the simulation.
        headless: Flag indicating whether the simulation should be run in headless mode.

    Attributes:
        last_feet_z (float): The z-coordinate of the last feet position.
        feet_height (torch.Tensor): Tensor representing the height of the feet.
        sim (gymtorch.GymSim): The simulation object.
        terrain (HumanoidTerrain): The terrain object.
        up_axis_idx (int): The index representing the up axis.
        command_input (torch.Tensor): Tensor representing the command input.
        privileged_obs_buf (torch.Tensor): Tensor representing the privileged observations buffer.
        obs_buf (torch.Tensor): Tensor representing the observations buffer.
        obs_history (collections.deque): Deque containing the history of observations.
        critic_history (collections.deque): Deque containing the history of critic observations.

    Methods:
        _push_robots(): Randomly pushes the robots by setting a randomized base velocity.
        create_sim(): Creates the simulation, terrain, and environments.
        _get_noise_scale_vec(cfg): Sets a vector used to scale the noise added to the observations.
        step(actions): Performs a simulation step with the given actions.
        compute_observations(): Computes the observations.
        reset_idx(env_ids): Resets the environment for the specified environment IDs.
    """

    cfg: StompyCfg

    def __init__(
        self,
        cfg: StompyCfg,
        sim_params: SimParams,
        physics_engine: SimType,
        sim_device: str,
        headless: bool,
    ) -> None:
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

        self.last_feet_z = 0.05
        self.feet_height = torch.zeros((self.num_envs, 2), device=self.device)
        self.reset_idx(torch.tensor(range(self.num_envs), device=self.device))
        self.compute_observations_()

    def _push_robots(self) -> None:
        # """Random pushes the robots. Emulates an impulse by setting a randomized base velocity."""
        # max_vel = self.cfg.domain_rand.max_push_vel_xy
        # max_push_angular = self.cfg.domain_rand.max_push_ang_vel

        # # Linear velocity in the X / Y axes.
        # self.rand_push_force[:, :2] = torch_rand_float(
        #     -max_vel,
        #     max_vel,
        #     (self.num_envs, 2),
        #     device=self.device,
        # )
        # self.root_states[:, 7:9] = self.rand_push_force[:, :2]

        # # Random torques in all three axes.
        # self.rand_push_torque = torch_rand_float(
        #     -max_push_angular,
        #     max_push_angular,
        #     (self.num_envs, 3),
        #     device=self.device,
        # )
        # self.root_states[:, 10:13] = self.rand_push_torque

        # self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

        pass

    def create_sim(self) -> None:
        """Creates simulation, terrain and evironments."""
        self.up_axis_idx = 2  # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(
            self.sim_device_id,
            self.graphics_device_id,
            self.physics_engine,
            self.sim_params,
        )
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ["heightfield", "trimesh"]:
            self.terrain = HumanoidTerrain(self.cfg.terrain, self.num_envs)
        if mesh_type == "plane":
            self._create_ground_plane()
        elif mesh_type == "heightfield":
            self._create_heightfield()
        elif mesh_type == "trimesh":
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError("Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
        self._create_envs()

    def _get_noise_scale_vec(self, cfg: StompyCfg) -> Tensor:
        """Sets a vector used to scale the noise added to the observations.

        Args:
            cfg: The environment config file.

        Returns:
            Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros(cfg.env.num_single_obs, device=self.device)
        self.add_noise = cfg.noise.add_noise
        noise_scales = cfg.noise.noise_scales
        noise_vec[0:12] = noise_scales.dof_pos * self.obs_scales.dof_pos
        noise_vec[12:24] = noise_scales.dof_vel * self.obs_scales.dof_vel
        noise_vec[24:36] = 0.0  # Previous actions
        noise_vec[36:39] = noise_scales.ang_vel * self.obs_scales.ang_vel  # Angular velocity
        noise_vec[39:42] = noise_scales.quat * self.obs_scales.quat  # Gyroscope (x, y)
        return noise_vec

    def step(self, actions: Tensor) -> Tensor:
        # Dynamic randomization. We simulate a delay in the actions to
        # help the policy generalize better in the real world.
        delay = torch.rand((self.num_envs, 1), device=self.device)
        actions = (1 - delay) * actions + delay * self.actions
        actions += self.cfg.domain_rand.dynamic_randomization * torch.randn_like(actions) * actions
        return super().step(actions)

    def compute_observations_(self) -> None:
        contact_mask = self.contact_forces[:, self.feet_indices, 2] > 5.0

        self.command_input = torch.cat((self.commands[:, :3] * self.commands_scale), dim=1)

        q = (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos
        dq = self.dof_vel * self.obs_scales.dof_vel

        diff = self.dof_pos

        self.privileged_obs_buf = torch.cat(
            (
                self.command_input,  # 2 + 3
                (self.dof_pos - self.default_joint_pd_target) * self.obs_scales.dof_pos,  # 12
                self.dof_vel * self.obs_scales.dof_vel,  # 12
                self.actions,  # 12
                diff,  # 12
                self.base_lin_vel * self.obs_scales.lin_vel,  # 3
                self.base_ang_vel * self.obs_scales.ang_vel,  # 3
                self.base_euler_xyz * self.obs_scales.quat,  # 3
                self.rand_push_force[:, :2],  # 3
                self.rand_push_torque,  # 3
                self.env_frictions,  # 1
                self.body_mass / 30.0,  # 1
                contact_mask,  # 2
            ),
            dim=-1,
        )

        obs_buf = torch.cat(
            (
                self.command_input,  # 5 = 2D(sin cos) + 3D(vel_x, vel_y, aug_vel_yaw)
                q,  # 12D
                dq,  # 12D
                self.actions,  # 12D
                self.base_ang_vel * self.obs_scales.ang_vel,  # 3
                self.base_euler_xyz * self.obs_scales.quat,  # 3
            ),
            dim=-1,
        )

        if self.cfg.terrain.measure_heights:
            heights = (
                torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.0)
                * self.obs_scales.height_measurements
            )
            self.privileged_obs_buf = torch.cat((self.obs_buf, heights), dim=-1)

        if self.add_noise:
            obs_now = obs_buf.clone() + torch.randn_like(obs_buf) * self.noise_scale_vec * self.cfg.noise.noise_level
        else:
            obs_now = obs_buf.clone()
        self.obs_history.append(obs_now)
        self.critic_history.append(self.privileged_obs_buf)

        obs_buf_all = torch.stack([self.obs_history[i] for i in range(self.obs_history.maxlen)], dim=1)  # N,T,K

        self.obs_buf = obs_buf_all.reshape(self.num_envs, -1)  # N, T*K
        self.privileged_obs_buf = torch.cat([self.critic_history[i] for i in range(self.cfg.env.c_frame_stack)], dim=1)

    def reset_idx(self, env_ids: int) -> None:
        super().reset_idx(env_ids)

        for i in range(self.obs_history.maxlen):
            self.obs_history[i][env_ids] *= 0
        for i in range(self.critic_history.maxlen):
            self.critic_history[i][env_ids] *= 0

    def _reward_feet_distance(self) -> Tensor:
        """Calculates the reward based on the distance between the feet.

        Returns:
            The reward for keeping the feet within a specified distance range.
        """
        foot_pos = self.rigid_state[:, self.feet_indices, :2]
        foot_dist = torch.norm(foot_pos[:, 0, :] - foot_pos[:, 1, :], dim=1)
        fd = self.cfg.rewards.min_dist
        max_df = self.cfg.rewards.max_dist
        d_min = torch.clamp(foot_dist - fd, -0.5, 0.0)
        d_max = torch.clamp(foot_dist - max_df, 0, 0.5)
        return (torch.exp(-torch.abs(d_min) * 100) + torch.exp(-torch.abs(d_max) * 100)) / 2

    def _reward_knee_distance(self) -> Tensor:
        """Calculates the reward based on the distance between the knee of the humanoid.

        Returns:
            The reward for keeping the knees within a specified range.
        """
        foot_pos = self.rigid_state[:, self.knee_indices, :2]
        foot_dist = torch.norm(foot_pos[:, 0, :] - foot_pos[:, 1, :], dim=1)
        fd = self.cfg.rewards.min_dist
        max_df = self.cfg.rewards.max_dist / 2
        d_min = torch.clamp(foot_dist - fd, -0.5, 0.0)
        d_max = torch.clamp(foot_dist - max_df, 0, 0.5)
        return (torch.exp(-torch.abs(d_min) * 100) + torch.exp(-torch.abs(d_max) * 100)) / 2

    def _reward_foot_slip(self) -> Tensor:
        """Calculates the reward for minimizing foot slip.

        The reward is based on the contact forces and the speed of the feet.
        A contact threshold is used to determine if the foot is in contact with
        the ground. The speed of the foot is calculated and scaled by the
        contact condition.

        Returns:
            The reward for minimizing foot slip.
        """
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.0
        foot_speed_norm = torch.norm(self.rigid_state[:, self.feet_indices, 10:12], dim=2)
        rew = torch.sqrt(foot_speed_norm)
        rew *= contact
        return torch.sum(rew, dim=1)

    def _reward_feet_air_time(self) -> Tensor:
        """Calculates the reward for feet air time, promoting longer steps.

        This is achieved by checking the first contact with the ground after
        being in the air. The air time is limited to a maximum value for
        reward calculation.

        Returns:
            The reward for maximizing the time the feet are in the air.
        """
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.0
        stance_mask = self._get_gait_phase()
        self.contact_filt = torch.logical_or(torch.logical_or(contact, stance_mask), self.last_contacts)
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.0) * self.contact_filt
        self.feet_air_time += self.dt
        air_time = self.feet_air_time.clamp(0, 0.5) * first_contact
        self.feet_air_time *= ~self.contact_filt
        return air_time.sum(dim=1)

    def _reward_feet_contact_number(self) -> Tensor:
        """Calculates a reward based on the number of feet contacts.

        Returns:
            Reward or penalty encouraging the contact number to match the
            expected gait phase.
        """
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.0
        stance_mask = self._get_gait_phase()
        reward = torch.where(contact == stance_mask, 1, -0.3)
        return torch.mean(reward, dim=1)

    def _reward_orientation(self) -> Tensor:
        """Calculates the reward for maintaining a flat base orientation.

        It penalizes deviation from the desired base orientation using the
        base euler angles and the projected gravity vector.

        Returns:
            The reward for maintaining a flat base orientation.
        """
        quat_mismatch = torch.exp(-torch.sum(torch.abs(self.base_euler_xyz[:, :2]), dim=1) * 10)
        orientation = torch.exp(-torch.norm(self.projected_gravity[:, :2], dim=1) * 20)
        return (quat_mismatch + orientation) / 2.0

    def _reward_feet_contact_forces(self) -> Tensor:
        """Calculates the reward for keeping contact forces within a range.

        Penalizes high contact forces on the feet.

        Returns:
            The reward for keeping contact forces within a specified range.
        """
        return torch.sum(
            (
                torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) - self.cfg.rewards.max_contact_force
            ).clip(0, 400),
            dim=1,
        )

    def _reward_default_joint_pos(self) -> Tensor:
        """Calculates the reward for matching the target position.

        Calculates the reward for keeping joint positions close to default
        positions, with a focus on penalizing deviation in yaw and roll
        directions. Excludes yaw and roll from the main penalty.

        Returns:
            The reward for matching the target joint position.
        """
        joint_diff = self.dof_pos - self.default_joint_pd_target
        left_yaw_roll = joint_diff[:, :2]
        right_yaw_roll = joint_diff[:, 6:8]
        yaw_roll = torch.norm(left_yaw_roll, dim=1) + torch.norm(right_yaw_roll, dim=1)
        yaw_roll = torch.clamp(yaw_roll - 0.1, 0, 50)
        return torch.exp(-yaw_roll * 100) - 0.01 * torch.norm(joint_diff, dim=1)

    def _reward_base_height(self) -> Tensor:
        """Calculates the reward based on the robot's base height.

        Penalizes deviation from a target base height. The reward is computed
        based on the height difference between the robot's base and the average
        height of its feet when they are in contact with the ground.

        Returns:
            The reward for maintaining the robot's base height.
        """
        stance_mask = self._get_gait_phase()
        height_num = torch.sum(self.rigid_state[:, self.feet_indices, 2] * stance_mask, dim=1)
        height_denom = torch.sum(stance_mask, dim=1)
        measured_heights = height_num / height_denom
        base_height = self.root_states[:, 2] - (measured_heights - 0.05)
        return torch.exp(-torch.abs(base_height - self.cfg.rewards.base_height_target) * 100)

    def _reward_base_acc(self) -> Tensor:
        """Computes the reward based on the base's acceleration.

        Penalizes high accelerations of the robot's base, encouraging
        smoother motion.

        Returns:
            The reward for maintaining the robot's base acceleration.
        """
        root_acc = self.last_root_vel - self.root_states[:, 7:13]
        rew = torch.exp(-torch.norm(root_acc, dim=1) * 3)
        return rew

    def _reward_vel_mismatch_exp(self) -> Tensor:
        """Computes a reward matching linear and angular velocities.

        Encourages the robot to maintain a stable velocity by penalizing
        large deviations.

        Returns:
            The reward for minimizing the mismatch in linear and angular velocities.
        """
        lin_mismatch = torch.exp(-torch.square(self.base_lin_vel[:, 2]) * 10)
        ang_mismatch = torch.exp(-torch.norm(self.base_ang_vel[:, :2], dim=1) * 5.0)

        c_update = (lin_mismatch + ang_mismatch) / 2.0

        return c_update

    def _reward_track_vel_hard(self) -> Tensor:
        """Encourages the robot to match linear and angular velocity commands.

        Penalizes deviations from specified linear and angular velocity targets.

        Returns:
            The reward for matching linear and angular velocity commands.
        """
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.norm(self.commands[:, :2] - self.base_lin_vel[:, :2], dim=1)
        lin_vel_error_exp = torch.exp(-lin_vel_error * 10)

        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.abs(self.commands[:, 2] - self.base_ang_vel[:, 2])
        ang_vel_error_exp = torch.exp(-ang_vel_error * 10)

        linear_error = 0.2 * (lin_vel_error + ang_vel_error)

        return (lin_vel_error_exp + ang_vel_error_exp) / 2.0 - linear_error

    def _reward_tracking_lin_vel(self) -> Tensor:
        """Tracks linear velocity commands along the ``x`` and ``y`` axes.

        Calculates a reward based on how closely the robot's linear velocity
        matches the commanded values.

        Returns:
            The reward for tracking linear velocity commands.
        """
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error * self.cfg.rewards.tracking_sigma)

    def _reward_tracking_ang_vel(self) -> Tensor:
        """Tracks angular velocity commands for yaw rotation.

        Computes a reward based on how closely the robot's angular velocity
        matches the commanded yaw values.

        Returns:
            The reward for tracking angular velocity commands.
        """
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error * self.cfg.rewards.tracking_sigma)

    def _reward_feet_clearance(self) -> Tensor:
        """Calculates reward for the leg swing above the ground.

        Encourages appropriate lift of the feet during the swing phase of the
        gait.

        Returns:
            The reward for maintaining the feet clearance.
        """
        # Compute feet contact mask
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.0

        # Get the z-position of the feet and compute the change in z-position
        feet_z = self.rigid_state[:, self.feet_indices, 2] - 0.05
        delta_z = feet_z - self.last_feet_z
        self.feet_height += delta_z
        self.last_feet_z = feet_z

        # Compute swing mask
        swing_mask = 1 - self._get_gait_phase()

        # Feet height should be closed to target feet height at the peak
        rew_pos = torch.abs(self.feet_height - self.cfg.rewards.target_feet_height) < 0.01
        rew_pos = torch.sum(rew_pos * swing_mask, dim=1)
        self.feet_height *= ~contact

        return rew_pos

    def _reward_low_speed(self) -> Tensor:
        """Calculates reward to match the command speed.

        This function checks if the robot is moving too slow, too fast, or at
        the desired speed, and if the movement direction matches the command.

        Returns:
            The reward for matching the command speed.
        """
        # Calculate the absolute value of speed and command for comparison
        absolute_speed = torch.abs(self.base_lin_vel[:, 0])
        absolute_command = torch.abs(self.commands[:, 0])

        # Define speed criteria for desired range
        speed_too_low = absolute_speed < 0.5 * absolute_command
        speed_too_high = absolute_speed > 1.2 * absolute_command
        speed_desired = ~(speed_too_low | speed_too_high)

        # Check if the speed and command directions are mismatched
        sign_mismatch = torch.sign(self.base_lin_vel[:, 0]) != torch.sign(self.commands[:, 0])

        # Initialize reward tensor
        reward = torch.zeros_like(self.base_lin_vel[:, 0])

        # Speed too low
        reward[speed_too_low] = -1.0
        # Speed too high
        reward[speed_too_high] = 0.0
        # Speed within desired range
        reward[speed_desired] = 1.2
        # Sign mismatch has the highest priority
        reward[sign_mismatch] = -2.0

        return reward * (self.commands[:, 0].abs() > 0.1)

    def _reward_torques(self) -> Tensor:
        """Penalizes the use of high torques in the robot's joints.

        Encourages efficient movement by minimizing the necessary force exerted
        by the motors.

        Returns:
            The reward for minimizing the use of high torques.
        """
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_vel(self) -> Tensor:
        """Penalizes high velocities at the degrees of freedom of the robot.

        This encourages smoother and more controlled movements.

        Returns:
            The reward for minimizing high velocities at the robot's degrees
            of freedom.
        """
        return torch.sum(torch.square(self.dof_vel), dim=1)

    def _reward_dof_acc(self) -> Tensor:
        """Penalizes high accelerations at the degrees of freedom of the robot.

        This is important for ensuring smooth and stable motion, reducing
        wear on the robot's mechanical parts.

        Returns:
            The reward for minimizing high accelerations at the robot's degrees
            of freedom.
        """
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)

    def _reward_collision(self) -> Tensor:
        """Penalizes collisions of the robot with the environment.

        This specifically focuses on selected body parts. This encourages the
        robot to avoid undesired contact with objects or surfaces.

        Returns:
            The reward for minimizing collisions.
        """
        return torch.sum(
            1.0 * (torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1
        )

    def _reward_action_smoothness(self) -> Tensor:
        """Penalizes large differences between consecutive actions.

        This is important for achieving fluid motion and reducing mechanical stress.

        Returns:
            The reward for minimizing differences between consecutive actions.
        """
        term_1 = torch.sum(torch.square(self.last_actions - self.actions), dim=1)
        term_2 = torch.sum(torch.square(self.actions + self.last_last_actions - 2 * self.last_actions), dim=1)
        term_3 = 0.05 * torch.sum(torch.abs(self.actions), dim=1)
        return term_1 + term_2 + term_3
