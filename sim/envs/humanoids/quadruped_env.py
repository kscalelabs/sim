# mypy: disable-error-code="valid-newtype"
"""Defines the environment for training the humanoid."""

from sim.envs.base.quadruped_robot import LeggedRobot
from sim.resources.quadruped.joints import Robot
from sim.utils.terrain import HumanoidTerrain

from isaacgym import gymtorch  # isort:skip
from isaacgym.torch_utils import *  # isort: skip


import torch  # isort:skip


class QuadrupedFreeEnv(LeggedRobot):
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
        self.feet_height = torch.zeros((self.num_envs, 4), device=self.device)
        self.reset_idx(torch.tensor(range(self.num_envs), device=self.device))

        env_handle = self.envs[0]
        actor_handle = self.actor_handles[0]

        self.legs_joints = {}
        for name, joint in Robot.legs.left.joints_motors():
            print(name)
            joint_handle = self.gym.find_actor_dof_handle(env_handle, actor_handle, joint)
            self.legs_joints["left_" + name] = joint_handle

        for name, joint in Robot.legs.right.joints_motors():
            joint_handle = self.gym.find_actor_dof_handle(env_handle, actor_handle, joint)
            self.legs_joints["right_" + name] = joint_handle

        # For quadruped, treat "arms" as legs
        for name, joint in Robot.arms.left.joints_motors():
            print(name)
            joint_handle = self.gym.find_actor_dof_handle(env_handle, actor_handle, joint)
            self.legs_joints["left_" + name] = joint_handle

        for name, joint in Robot.arms.right.joints_motors():
            joint_handle = self.gym.find_actor_dof_handle(env_handle, actor_handle, joint)
            self.legs_joints["right_" + name] = joint_handle

        self.compute_observations()

    def _push_robots(self):
        """Random pushes the robots. Emulates an impulse by setting a randomized base velocity."""
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        max_push_angular = self.cfg.domain_rand.max_push_ang_vel
        self.rand_push_force[:, :2] = torch_rand_float(
            -max_vel, max_vel, (self.num_envs, 2), device=self.device
        )  # lin vel x/y
        self.root_states[:, 7:9] = self.rand_push_force[:, :2]

        self.rand_push_torque = torch_rand_float(
            -max_push_angular, max_push_angular, (self.num_envs, 3), device=self.device
        )
        self.root_states[:, 10:13] = self.rand_push_torque

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
        stance_mask = torch.zeros((self.num_envs, 4), device=self.device)
        # left foot stance
        stance_mask[:, 0] = sin_pos >= 0
        stance_mask[:, 3] = sin_pos >= 0
        # right foot stance
        stance_mask[:, 1] = sin_pos < 0
        stance_mask[:, 2] = sin_pos < 0
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
        self.ref_dof_pos[:, self.legs_joints["left_hip_pitch"]] += sin_pos_l * scale_1
        self.ref_dof_pos[:, self.legs_joints["left_knee_pitch"]] += sin_pos_l * scale_2
        self.ref_dof_pos[:, self.legs_joints["left_elbow_pitch"]] += sin_pos_l * scale_1
        self.ref_dof_pos[:, self.legs_joints["left_shoulder_pitch"]] += sin_pos_l * scale_2
        # right foot stance phase set to default joint pos
        sin_pos_r[sin_pos_r < 0] = 0
        self.ref_dof_pos[:, self.legs_joints["right_hip_pitch"]] += sin_pos_r * scale_1
        self.ref_dof_pos[:, self.legs_joints["right_knee_pitch"]] += sin_pos_r * scale_2
        self.ref_dof_pos[:, self.legs_joints["right_elbow_pitch"]] += sin_pos_r * scale_1
        self.ref_dof_pos[:, self.legs_joints["right_shoulder_pitch"]] += sin_pos_r * scale_2

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
        num_actions = self.num_actions
        noise_vec = torch.zeros(self.cfg.env.num_single_obs, device=self.device)
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_vec[0:5] = 0.0  # commands
        noise_vec[5 : (num_actions + 5)] = noise_scales.dof_pos * self.obs_scales.dof_pos
        noise_vec[(num_actions + 5) : (2 * num_actions + 5)] = noise_scales.dof_vel * self.obs_scales.dof_vel
        noise_vec[(2 * num_actions + 5) : (3 * num_actions + 5)] = 0.0  # previous actions
        noise_vec[(3 * num_actions + 5) : (3 * num_actions + 5) + 3] = (
            noise_scales.ang_vel * self.obs_scales.ang_vel
        )  # ang vel
        noise_vec[(3 * num_actions + 5) + 3 : (3 * num_actions + 5)] = (
            noise_scales.quat * self.obs_scales.quat
        )  # euler x,y
        return noise_vec

    def step(self, actions):
        if self.cfg.env.use_ref_actions:
            actions += self.ref_action
        # dynamic randomization
        delay = torch.rand((self.num_envs, 1), device=self.device)
        actions = (1 - delay) * actions + delay * self.actions
        actions += self.cfg.domain_rand.dynamic_randomization * torch.randn_like(actions) * actions
        return super().step(actions)

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
                q,  # 20D
                dq,  # 20D
                self.actions,  # 20D
                self.base_ang_vel * self.obs_scales.ang_vel,  # 3
                self.base_euler_xyz * self.obs_scales.quat,  # 3
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
    # changed from unitree repo
    def _reward_orientation(self):
        quat_mismatch = torch.exp(-torch.sum(torch.abs(self.base_euler_xyz[:, :2]), dim=1) * 15)
        orientation = torch.exp(-torch.norm(self.projected_gravity[:, :2], dim=1) * 20)
        return (quat_mismatch + orientation) / 2

    # this function was not in unitree repo, added from humanoids repo
    def _reward_default_joint_pos(self):
        """Calculates the reward for keeping joint positions close to default positions
        Should
        """
        joint_diff = self.dof_pos - self.default_joint_pd_target

        left_pitch = joint_diff[
            :,
            [
                self.legs_joints["left_hip_pitch"],
                self.legs_joints["left_knee_pitch"],
                self.legs_joints["left_elbow_pitch"],
                self.legs_joints["left_shoulder_pitch"],
            ],
        ]
        right_pitch = joint_diff[
            :,
            [
                self.legs_joints["right_hip_pitch"],
                self.legs_joints["right_knee_pitch"],
                self.legs_joints["right_elbow_pitch"],
                self.legs_joints["right_shoulder_pitch"],
            ],
        ]
        pitch_dev = torch.norm(left_pitch, dim=1) + torch.norm(right_pitch, dim=1)
        pitch_dev = torch.clamp(pitch_dev - 0.1, 0, 50)  # deadzone of 0.1, max 50 min 0
        return torch.exp(-pitch_dev * 0.5) - 0.01 * torch.norm(joint_diff, dim=1)  #

    # this function was not in unitree repo, added from humanoids repo
    def _reward_base_acc(self):
        """Computes the reward based on the base's acceleration. Penalizes high accelerations of the robot's base,
        encouraging smoother motion.
        """
        root_acc = self.last_root_vel - self.root_states[:, 7:13]
        rew = torch.exp(-torch.norm(root_acc, dim=1) * 3)
        return rew

    # FOR WALKING:
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    # Changed from unitree repo
    def _reward_base_height(self):
        stance_mask = self._get_gait_phase()
        measured_heights = torch.sum(self.rigid_state[:, self.feet_indices, 2] * stance_mask, dim=1) / torch.sum(
            stance_mask, dim=1
        )
        base_height = self.root_states[:, 2] - (measured_heights - self.cfg.asset.default_feet_height)
        reward = torch.exp(-torch.abs(base_height - self.cfg.rewards.base_height_target) * 10)
        return reward

    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel), dim=1)

    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)

    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(
            1.0 * (torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1
        )

    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf

    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.0)  # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.0)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum(
            (torch.abs(self.dof_vel) - self.dof_vel_limits * self.cfg.rewards.soft_dof_vel_limit).clip(
                min=0.0, max=1.0
            ),
            dim=1,
        )

    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum(
            (torch.abs(self.torques) - self.torque_limits * self.cfg.rewards.soft_torque_limit).clip(min=0.0), dim=1
        )

    # the main walking param
    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(
            torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1
        )  # around 11 max error
        return torch.exp(-(lin_vel_error * self.cfg.rewards.tracking_sigma * (1 / 10)))  # tracking_sigma = 4

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])  # around 1 max error
        return torch.exp(-(ang_vel_error * self.cfg.rewards.tracking_sigma))

    def _reward_feet_air_time(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.0
        stance_mask = self._get_gait_phase()
        self.contact_filt = torch.logical_or(torch.logical_or(contact, stance_mask), self.last_contacts)
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.0) * self.contact_filt
        self.feet_air_time += self.dt
        air_time = self.feet_air_time.clamp(0, 0.5) * first_contact
        self.feet_air_time *= ~self.contact_filt
        return air_time.sum(dim=1)

    def _reward_stumble(self):
        # Penalize feet hitting vertical surfaces
        return torch.any(
            torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2)
            > 5 * torch.abs(self.contact_forces[:, self.feet_indices, 2]),
            dim=1,
        )

    def _reward_stand_still(self):
        # Penalize motion at zero commands
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * (
            torch.norm(self.commands[:, :2], dim=1) < 0.1
        )

    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum(
            (
                torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) - self.cfg.rewards.max_contact_force
            ).clip(min=0.0),
            dim=1,
        )

    # #Added from humanoids

    # def _reward_low_speed(self):
    #     """Rewards or penalizes the robot based on its speed relative to the commanded speed.
    #     This function checks if the robot is moving too slow, too fast, or at the desired speed,
    #     and if the movement direction matches the command.
    #     """
    #     # Calculate the absolute value of speed and command for comparison
    #     absolute_speed = torch.abs(self.base_lin_vel[:, 0])
    #     absolute_command = torch.abs(self.commands[:, 0])
    #     # Define speed criteria for desired range
    #     speed_too_low = absolute_speed < 0.5 * absolute_command
    #     speed_too_high = absolute_speed > 1.2 * absolute_command
    #     speed_desired = ~(speed_too_low | speed_too_high)

    #     # Check if the speed and command directions are mismatched
    #     sign_mismatch = torch.sign(self.base_lin_vel[:, 0]) != torch.sign(self.commands[:, 0])

    #     # Initialize reward tensor
    #     reward = torch.zeros_like(self.base_lin_vel[:, 0])

    #     # Assign rewards based on conditions
    #     # Speed too low
    #     reward[speed_too_low] = 1.0 #-1.0
    #     # Speed too high
    #     reward[speed_too_high] = 1.5 #0.0
    #     # Speed within desired range
    #     reward[speed_desired] = 3.0 #1.2
    #     # Sign mismatch has the highest priority
    #     reward[sign_mismatch] = 0.0 #-2.0

    #     return reward * (self.commands[:, 0].abs() > 0.1)

    # #Added from humanoids
    # def _reward_feet_clearance(self):
    #     """Calculates reward based on the clearance of the swing leg from the ground during movement.
    #     Encourages appropriate lift of the feet during the swing phase of the gait.
    #     """
    #     # Compute feet contact mask
    #     contact = self.contact_forces[:, self.feet_indices, 2] > 5.0

    #     # Get the z-position of the feet and compute the change in z-position
    #     feet_z = self.rigid_state[:, self.feet_indices, 2] - self.cfg.asset.default_feet_height
    #     delta_z = feet_z - self.last_feet_z
    #     self.feet_height += delta_z
    #     self.last_feet_z = feet_z

    #     # Compute swing mask
    #     swing_mask = 1 - self._get_gait_phase()

    #     # feet height should be closed to target feet height at the peak
    #     rew_pos = torch.abs(self.feet_height - self.cfg.rewards.target_feet_height) < 0.02
    #     rew_pos = torch.sum(rew_pos * swing_mask, dim=1)
    #     self.feet_height *= ~contact

    #     return rew_pos
