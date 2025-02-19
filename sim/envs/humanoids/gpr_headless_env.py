# mypy: disable-error-code="valid-newtype"
"""Defines the environment for training the humanoid."""

from isaacgym.torch_utils import *  # isort:skip

from sim.envs.base.legged_robot import LeggedRobot
from sim.resources.gpr_headless.joints import Robot
from sim.utils.terrain import HumanoidTerrain
from sim.utils.math import wrap_to_pi


from isaacgym import gymtorch  # isort:skip

import torch  # isort:skip
import random  # isort:skip


class GprHeadlessEnv(LeggedRobot):
    """GprHeadlessEnv is a class that represents a custom environment for a legged robot.

    Args:
        cfg (LeggedRobotCfg): Configuration object for the legged robot.
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
        _get_phase(): Calculates the phase of the gait cycle.
        _get_gait_phase(): Calculates the gait phase.
        compute_ref_state(): Computes the reference state.
        create_sim(): Creates the simulation, terrain, and environments.
        _get_noise_scale_vec(cfg): Sets a vector used to scale the noise added to the observations.
        step(actions): Performs a simulation step with the given actions.
        compute_observations(): Computes the observations.
        reset_idx(env_ids): Resets the environment for the specified environment IDs.
    """

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
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

        self.compute_observations()
        self._initialize_push_intervals()

    def step(self, actions):
        """Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        if self.cfg.env.use_ref_actions:
            actions += self.ref_action
        actions = torch.clip(actions, -self.cfg.normalization.clip_actions, self.cfg.normalization.clip_actions)

        # dynamic randomization
        delay = torch.rand((self.num_envs, 1), device=self.device) * self.cfg.domain_rand.action_delay
        actions = (1 - delay) * actions + delay * self.actions
        actions += self.cfg.domain_rand.action_noise * torch.randn_like(actions) * actions

        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        # step physics and render each frame
        target_positions = self.actions[:,:self.cfg.env.num_actions//2]
        target_velocities = self.actions[:,self.cfg.env.num_actions//2:]
        self.render()
        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(target_positions, target_velocities).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))

            self.gym.simulate(self.sim)
            if self.device == "cpu":
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def _compute_torques(self, positions, velocities):
        """Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        pos_scaled = positions * self.cfg.control.action_scale
        vel_scaled = velocities * self.cfg.control.action_scale
        p_gains = self.p_gains
        d_gains = self.d_gains
        torques = p_gains * (pos_scaled + self.default_dof_pos - self.dof_pos) + d_gains * (vel_scaled - self.dof_vel)
        res = torch.clip(torques, -self.torque_limits, self.torque_limits)
        return res

    def _push_robots(self, env_ids=None):
        """Randomly pushes the robots for specific environments.
        
        If env_ids is None, push all environments; otherwise, push only for the provided indices.
        """
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        max_push_angular = self.cfg.domain_rand.max_push_ang_vel
        
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        
        # For the selected environments, sample random push forces and torques
        self.rand_push_force[env_ids, :2] = torch_rand_float(
            -max_vel, max_vel, (len(env_ids), 2), device=self.device
        )
        self.root_states[env_ids, 7:9] = self.rand_push_force[env_ids, :2]
        
        self.rand_push_torque[env_ids] = torch_rand_float(
            -max_push_angular, max_push_angular, (len(env_ids), 3), device=self.device
        )
        self.root_states[env_ids, 10:13] = self.rand_push_torque[env_ids]

        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    def _get_phase(self):
        cycle_time = self.cfg.rewards.cycle_time
        phase = self.episode_length_buf * self.dt / cycle_time
        return phase

    def _get_gait_phase(self):
        # return float mask 1 is stance, 0 is swing
        phase = self._get_phase()
        sin_pos = torch.sin(2 * torch.pi * phase)
        # Add double support phase
        stance_mask = torch.zeros((self.num_envs, 2), device=self.device)
        # left foot stance
        stance_mask[:, 0] = sin_pos >= 0
        # right foot stance
        stance_mask[:, 1] = sin_pos < 0
        # Double support phase
        stance_mask[torch.abs(sin_pos) < 0.1] = 1

        return stance_mask

    def check_termination(self):
        """Check if environments need to be reset"""
        self.reset_buf = torch.any(
            torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1.0,
            dim=1,
        )
        self.reset_buf |= self.root_states[:, 2] < self.cfg.asset.termination_height
        self.time_out_buf = self.episode_length_buf > self.max_episode_length  # no terminal reward for time-outs
        self.reset_buf |= self.time_out_buf

    def compute_ref_state(self):
        phase = self._get_phase()
        sin_pos = torch.sin(2 * torch.pi * phase)
        sin_pos_l = sin_pos.clone()
        sin_pos_r = sin_pos.clone()
        default_clone = self.default_dof_pos.clone()
        self.ref_dof_pos = self.default_dof_pos.repeat(self.num_envs, 1)

        scale_1 = self.cfg.rewards.target_joint_pos_scale
        scale_2 = 2 * scale_1
        # left foot stance phase set to default joint pos
        sin_pos_l[sin_pos_l > 0] = 0
   
        self.ref_dof_pos[:, self.legs_joints["left_hip_pitch"]] += -sin_pos_l * scale_1
        self.ref_dof_pos[:, self.legs_joints["left_knee_pitch"]] += -sin_pos_l * scale_2
        self.ref_dof_pos[:, self.legs_joints["left_ankle_pitch"]] += sin_pos_l * scale_1

        # right foot stance phase set to default joint pos
        sin_pos_r[sin_pos_r < 0] = 0
        # pfb30 last one should be -1
        self.ref_dof_pos[:, self.legs_joints["right_hip_pitch"]] += -sin_pos_r * scale_1
        self.ref_dof_pos[:, self.legs_joints["right_knee_pitch"]] += -sin_pos_r * scale_2
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

    def _get_noise_scale_vec(self, cfg):
        """Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        num_joints = self.num_actions // 2  # Since num_actions is now NUM_JOINTS * 2
        noise_vec = torch.zeros(self.cfg.env.num_single_obs, device=self.device)
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_vec[0:5] = 0.0  # commands
        noise_vec[5:(num_joints + 5)] = noise_scales.dof_pos * self.obs_scales.dof_pos
        noise_vec[(num_joints + 5):(2 * num_joints + 5)] = noise_scales.dof_vel * self.obs_scales.dof_vel
        noise_vec[(2 * num_joints + 5):(4 * num_joints + 5)] = 0.0  # previous actions (now includes both pos and vel)
        noise_vec[(4 * num_joints + 5):(4 * num_joints + 8)] = noise_scales.quat * self.obs_scales.quat  # projected_gravity
        return noise_vec

    def compute_observations(self):
        phase = self._get_phase()
        self.compute_ref_state()

        sin_pos = torch.sin(2 * torch.pi * phase).unsqueeze(1)
        cos_pos = torch.cos(2 * torch.pi * phase).unsqueeze(1)

        stance_mask = self._get_gait_phase()
        contact_mask = self.contact_forces[:, self.feet_indices, 2] > 5.0

        self.command_input = torch.cat((sin_pos, cos_pos, self.commands[:, :3] * self.commands_scale), dim=1)
        q = (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos
        dq = self.dof_vel * self.obs_scales.dof_vel

        diff = self.dof_pos - self.ref_dof_pos
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
                stance_mask,  # 2
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
                self.projected_gravity,  # 3
            ),
            dim=-1,
        )

        if self.cfg.terrain.measure_heights:
            heights = (
                torch.clip(
                    self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights,
                    -1,
                    1.0,
                )
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

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        for i in range(self.obs_history.maxlen):
            self.obs_history[i][env_ids] *= 0
        for i in range(self.critic_history.maxlen):
            self.critic_history[i][env_ids] *= 0

    # ================================================ Rewards ================================================== #
    def _reward_joint_pos(self):
        """Calculates the reward based on the difference between the current joint positions and the target joint positions."""
        joint_pos = self.dof_pos.clone()
        pos_target = self.ref_dof_pos.clone()
        diff = joint_pos - pos_target
        r = torch.exp(-2 * torch.norm(diff, dim=1)) - 0.2 * torch.norm(diff, dim=1).clamp(0, 0.5)

        return r

    def _reward_feet_distance(self):
        """Calculates the reward based on the distance between the feet. Penilize feet get close to each other or too far away."""
        foot_pos = self.rigid_state[:, self.feet_indices, :2]
        foot_dist = torch.norm(foot_pos[:, 0, :] - foot_pos[:, 1, :], dim=1)
        fd = self.cfg.rewards.min_dist
        max_df = self.cfg.rewards.max_dist
        d_min = torch.clamp(foot_dist - fd, -0.5, 0.0)
        d_max = torch.clamp(foot_dist - max_df, 0, 0.5)
        return (torch.exp(-torch.abs(d_min) * 100) + torch.exp(-torch.abs(d_max) * 100)) / 2

    def _reward_knee_distance(self):
        """Calculates the reward based on the distance between the knee of the humanoid."""
        foot_pos = self.rigid_state[:, self.knee_indices, :2]
        foot_dist = torch.norm(foot_pos[:, 0, :] - foot_pos[:, 1, :], dim=1)
        fd = self.cfg.rewards.min_dist
        max_df = self.cfg.rewards.max_dist / 2
        d_min = torch.clamp(foot_dist - fd, -0.5, 0.0)
        d_max = torch.clamp(foot_dist - max_df, 0, 0.5)
        return (torch.exp(-torch.abs(d_min) * 100) + torch.exp(-torch.abs(d_max) * 100)) / 2

    def _reward_foot_slip(self):
        """Calculates the reward for minimizing foot slip. The reward is based on the contact forces
        and the speed of the feet. A contact threshold is used to determine if the foot is in contact
        with the ground. The speed of the foot is calculated and scaled by the contact condition.
        """
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.0
        foot_speed_norm = torch.norm(self.rigid_state[:, self.feet_indices, 7:9], dim=2)
        rew = torch.sqrt(foot_speed_norm)
        rew *= contact
        return torch.sum(rew, dim=1)

    def _reward_feet_air_time(self):
        """Calculates the reward for feet air time, promoting longer steps. This is achieved by
        checking the first contact with the ground after being in the air. The air time is
        limited to a maximum value for reward calculation.
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

    def _reward_feet_contact_number(self):
        """Calculates a reward based on the number of feet contacts aligning with the gait phase.
        Rewards or penalizes depending on whether the foot contact matches the expected gait phase.
        """
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.0
        stance_mask = self._get_gait_phase()
        reward = torch.where(contact == stance_mask, 1.0, -0.3)
        return torch.mean(reward, dim=1)

    def _reward_orientation(self):
        """Calculates the reward for maintaining a flat base orientation. It penalizes deviation
        from the desired base orientation using the base euler angles and the projected gravity vector.
        """
        quat_mismatch = torch.exp(-torch.sum(torch.abs(self.base_euler_xyz[:, :2]), dim=1) * 10)
        orientation = torch.exp(-torch.norm(self.projected_gravity[:, :2], dim=1) * 20)

        return (quat_mismatch + orientation) / 2.0

    def _reward_feet_contact_forces(self):
        """Calculates the reward for keeping contact forces within a specified range. Penalizes
        high contact forces on the feet.
        """
        return torch.sum(
            (
                torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) - self.cfg.rewards.max_contact_force
            ).clip(0, 400),
            dim=1,
        )

    def _reward_default_joint_pos(self):
        """Calculates the reward for keeping joint positions close to default positions, with a focus
        on penalizing deviation in yaw and roll directions. Excludes yaw and roll from the main penalty.
        """
        joint_diff = self.dof_pos - self.default_joint_pd_target
        left_yaw_roll = joint_diff[:, [self.legs_joints["left_hip_roll"], self.legs_joints["left_hip_yaw"]]]
        right_yaw_roll = joint_diff[:, [self.legs_joints["right_hip_roll"], self.legs_joints["right_hip_yaw"]]]
        yaw_roll = torch.norm(left_yaw_roll, dim=1) + torch.norm(right_yaw_roll, dim=1)
        yaw_roll = torch.clamp(yaw_roll - 0.1, 0, 50)

        return torch.exp(-yaw_roll * 100) - 0.01 * torch.norm(joint_diff, dim=1)

    def _reward_base_height(self):
        """Calculates the reward based on the robot's base height. Penalizes deviation from a target base height.
        The reward is computed based on the height difference between the robot's base and the average height
        of its feet when they are in contact with the ground.
        """
        stance_mask = self._get_gait_phase()
        measured_heights = torch.sum(self.rigid_state[:, self.feet_indices, 2] * stance_mask, dim=1) / torch.sum(
            stance_mask, dim=1
        )
        base_height = self.root_states[:, 2] - (measured_heights - self.cfg.asset.default_feet_height)
        reward = torch.exp(-torch.abs(base_height - self.cfg.rewards.base_height_target) * 100)
        return reward

    def _reward_base_acc(self):
        """Computes the reward based on the base's acceleration. Penalizes high accelerations of the robot's base,
        encouraging smoother motion.
        """
        root_acc = self.last_root_vel - self.root_states[:, 7:13]
        rew = torch.exp(-torch.norm(root_acc, dim=1) * 3)
        return rew

    def _reward_vel_mismatch_exp(self):
        """Computes a reward based on the mismatch in the robot's linear and angular velocities.
        Encourages the robot to maintain a stable velocity by penalizing large deviations.
        """
        lin_mismatch = torch.exp(-torch.square(self.base_lin_vel[:, 2]) * 10)
        ang_mismatch = torch.exp(-torch.norm(self.base_ang_vel[:, :2], dim=1) * 5.0)

        c_update = (lin_mismatch + ang_mismatch) / 2.0

        return c_update

    def _reward_track_vel_hard(self):
        """Calculates a reward for accurately tracking both linear and angular velocity commands.
        Penalizes deviations from specified linear and angular velocity targets.
        """
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.norm(self.commands[:, :2] - self.base_lin_vel[:, :2], dim=1)
        lin_vel_error_exp = torch.exp(-lin_vel_error * 10)

        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.abs(self.commands[:, 2] - self.base_ang_vel[:, 2])
        ang_vel_error_exp = torch.exp(-ang_vel_error * 10)

        linear_error = 0.2 * (lin_vel_error + ang_vel_error)

        return (lin_vel_error_exp + ang_vel_error_exp) / 2.0 - linear_error

    def _reward_tracking_lin_vel(self):
        """Tracks linear velocity commands along the xy axes.
        Calculates a reward based on how closely the robot's linear velocity matches the commanded values.
        """
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error * self.cfg.rewards.tracking_sigma)

    def _reward_tracking_ang_vel(self):
        """Tracks angular velocity commands for yaw rotation.
        Computes a reward based on how closely the robot's angular velocity matches the commanded yaw values.
        """
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error * self.cfg.rewards.tracking_sigma)

    def _reward_feet_clearance(self):
        """Calculates reward based on the clearance of the swing leg from the ground during movement.
        Encourages appropriate lift of the feet during the swing phase of the gait.
        """
        # Compute feet contact mask
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.0

        # Get the z-position of the feet and compute the change in z-position
        feet_z = self.rigid_state[:, self.feet_indices, 2] - self.cfg.asset.default_feet_height
        delta_z = feet_z - self.last_feet_z
        self.feet_height += delta_z
        self.last_feet_z = feet_z

        # Compute swing mask
        swing_mask = 1 - self._get_gait_phase()

        # feet height should be closed to target feet height at the peak
        rew_pos = torch.abs(self.feet_height - self.cfg.rewards.target_feet_height) < 0.02
        rew_pos = torch.sum(rew_pos * swing_mask, dim=1)
        self.feet_height *= ~contact

        return rew_pos

    def _reward_low_speed(self):
        """Rewards or penalizes the robot based on its speed relative to the commanded speed.
        This function checks if the robot is moving too slow, too fast, or at the desired speed,
        and if the movement direction matches the command.
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

        # Assign rewards based on conditions
        # Speed too low
        reward[speed_too_low] = -1.0
        # Speed too high
        reward[speed_too_high] = 0.0
        # Speed within desired range
        reward[speed_desired] = 1.2
        # Sign mismatch has the highest priority
        reward[sign_mismatch] = -2.0
        return reward * (self.commands[:, 0].abs() > 0.1)

    def _reward_torques(self):
        """Penalizes the use of high torques in the robot's joints. Encourages efficient movement by minimizing
        the necessary force exerted by the motors.
        """
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_vel(self):
        """Penalizes high velocities at the degrees of freedom (DOF) of the robot. This encourages smoother and
        more controlled movements.
        """
        return torch.sum(torch.square(self.dof_vel), dim=1)

    def _reward_dof_acc(self):
        """Penalizes high accelerations at the robot's degrees of freedom (DOF). This is important for ensuring
        smooth and stable motion, reducing wear on the robot's mechanical parts.
        """
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)

    def _reward_collision(self):
        """Penalizes collisions of the robot with the environment, specifically focusing on selected body parts.
        This encourages the robot to avoid undesired contact with objects or surfaces.
        """
        return torch.sum(
            1.0 * (torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1),
            dim=1,
        )

    def _reward_action_smoothness(self):
        """Encourages smoothness in the robot's actions by penalizing large differences between consecutive actions.
        This is important for achieving fluid motion and reducing mechanical stress.
        """
        term_1 = torch.sum(torch.square(self.last_actions - self.actions), dim=1)
        term_2 = torch.sum(
            torch.square(self.actions + self.last_last_actions - 2 * self.last_actions),
            dim=1,
        )
        term_3 = 0.05 * torch.sum(torch.abs(self.actions), dim=1)
        return term_1 + term_2 + term_3
    
    def _reward_termination(self):
        """Returns a penalty for each environment that is terminated."""
        penalty = 1.0
        # Apply the penalty element-wise: only environments that need reset get the penalty
        return penalty * self.reset_buf.float()

    def _initialize_push_intervals(self):
        # Create a tensor of size (num_envs,) for the next push step.
        # We'll assume self.num_envs is available and self.dt is the simulation time step.
        self.next_push_steps = torch.empty(self.num_envs, dtype=torch.int64, device=self.device)
        for i in range(self.num_envs):
            random_interval_sec = random.uniform(
                self.cfg.domain_rand.push_random_interval_min,
                self.cfg.domain_rand.push_random_interval_max
            )
            self.next_push_steps[i] = self.common_step_counter + int(random_interval_sec / self.dt)

    def _post_physics_step_callback(self):
        """Callback called before computing terminations, rewards, and observations."""
        env_ids = (
            (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt) == 0)
            .nonzero(as_tuple=False)
            .flatten()
        )
        self._resample_commands(env_ids)
        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(0.5 * wrap_to_pi(self.commands[:, 3] - heading), -1.0, 1.0)
        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()
        if self.cfg.domain_rand.push_robots:
            # Initialize per-env push intervals if not done yet.
            if not hasattr(self, 'next_push_steps'):
                self._initialize_push_intervals()
            # Determine which environments are due for a push
            envs_to_push = (self.common_step_counter >= self.next_push_steps).nonzero(as_tuple=False).flatten()
            if len(envs_to_push) > 0:
                # Apply push only to the environments that are due
                self._push_robots(envs_to_push)
                # For each pushed environment, set a new random push time
                for env_id in envs_to_push:
                    random_interval_sec = random.uniform(
                        self.cfg.domain_rand.push_random_interval_min,
                        self.cfg.domain_rand.push_random_interval_max
                    )
                    self.next_push_steps[env_id] = self.common_step_counter + int(random_interval_sec / self.dt)
