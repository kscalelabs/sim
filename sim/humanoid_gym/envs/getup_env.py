# mypy: disable-error-code="valid-newtype"
"""Defines the environment for training the humanoid."""

import torch
from humanoid.envs import LeggedRobot
from humanoid.envs.base.legged_robot_config import LeggedRobotCfg
from humanoid.utils.terrain import HumanoidTerrain
from isaacgym import gymtorch
from isaacgym.torch_utils import *

from sim.stompy2.joints import Stompy

default_feet_height = 0.0


class GetupFreeEnv(LeggedRobot):
    """GetupFreeEnv is a class that represents getting up env requirments.

    Note that the underlying XBot training code has some interesting
    conventions to be aware of:

    - The function `_prepare_reward_function` looks for all the methods on this
      class that start with `_reward_` and adds them to the reward function.
      Each reward function should return a tensor of rewards.
    - The state tensor is a 13-dimensional tensor whose elements are:
        - 3D origin
        - 4D quaternion of the base
        - 3D linear velocity
        - 3D angular velocity

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
        _get_phase(): Calculates the phase of the gait cycle.
        _get_gait_phase(): Calculates the gait phase.
        compute_ref_state(): Computes the reference state.
        create_sim(): Creates the simulation, terrain, and environments.
        _get_noise_scale_vec(cfg): Sets a vector used to scale the noise added to the observations.
        step(actions): Performs a simulation step with the given actions.
        compute_observations(): Computes the observations.
        reset_idx(env_ids): Resets the environment for the specified environment IDs.
    """

    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self.last_feet_z = default_feet_height
        self.feet_height = torch.zeros((self.num_envs, 2), device=self.device)
        self.reset_idx(torch.tensor(range(self.num_envs), device=self.device))

        env_handle = self.envs[0]
        actor_handle = self.actor_handles[0]

        self.ref_dof_pos = torch.zeros_like(self.dof_pos)
        for joint in Stompy.all_joints():
            joint_handle = self.gym.find_actor_dof_handle(env_handle, actor_handle, joint)
            if joint in Stompy.default_sitting():
                self.ref_dof_pos[:, joint_handle] = torch.tensor(Stompy.default_sitting()[joint])

        self.compute_observations()

        self.height_history = torch.zeros((self.num_envs, int(self.max_episode_length.item())), device=self.device)
        self.window_size = int(self.max_episode_length.item() * 0.2)

    def _push_robots(self):
        """Random pushes the robots. Emulates an impulse by setting a randomized base velocity."""
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        max_push_angular = self.cfg.domain_rand.max_push_ang_vel
        self.rand_push_force[:, :2] = torch_rand_float(
            -max_vel,
            max_vel,
            (self.num_envs, 2),
            device=self.device,
        )  # lin vel x/y
        self.root_states[:, 7:9] = self.rand_push_force[:, :2]

        self.rand_push_torque = torch_rand_float(
            -max_push_angular,
            max_push_angular,
            (self.num_envs, 3),
            device=self.device,
        )

        self.root_states[:, 10:13] = self.rand_push_torque

        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

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
        self.command_input = torch.zeros([self.num_envs, 5], device=self.device)

        q = (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos
        dq = self.dof_vel * self.obs_scales.dof_vel

        self.privileged_obs_buf = torch.cat(
            (
                self.command_input,  # 2 + 3
                (self.dof_pos - self.default_joint_pd_target) * self.obs_scales.dof_pos,  # 12
                self.dof_vel * self.obs_scales.dof_vel,  # 12
                self.actions,  # 12
                torch.zeros([self.num_envs, self.num_dof], device=self.device),  # 12
                self.base_lin_vel * self.obs_scales.lin_vel,  # 3
                self.base_ang_vel * self.obs_scales.ang_vel,  # 3
                self.base_euler_xyz * self.obs_scales.quat,  # 3
                self.rand_push_force[:, :2],  # 3
                self.rand_push_torque,  # 3
                self.env_frictions,  # 1
                self.body_mass / 30.0,  # 1
                torch.zeros([self.num_envs, 2], device=self.device),  # 2
                torch.zeros([self.num_envs, 2], device=self.device),  # 2
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
        r = torch.exp(-2 * torch.norm(diff, dim=1)) - 0.2 * torch.norm(diff, dim=1)
        return r

    def _reward_orientation(self):
        """Calculates the reward for maintaining a flat base orientation. It penalizes deviation
        from the desired base orientation using the base euler angles and the projected gravity vector.
        """
        quat_mismatch = torch.exp(-torch.sum(torch.abs(self.base_euler_xyz[:, :2]), dim=1) * 10)
        orientation = torch.exp(-torch.norm(self.projected_gravity[:, :2], dim=1) * 20)

        return (quat_mismatch + orientation) / 2.0

    def _reward_default_joint_pos(self):
        """Calculates the reward for keeping joint positions close to default positions, with a focus
        on penalizing deviation in yaw and roll directions. Excludes yaw and roll from the main penalty.
        """
        joint_diff = self.dof_pos - self.default_joint_pd_target
        left_yaw_roll = joint_diff[:, :2]
        right_yaw_roll = joint_diff[:, 6:8]
        yaw_roll = torch.norm(left_yaw_roll, dim=1) + torch.norm(right_yaw_roll, dim=1)
        yaw_roll = torch.clamp(yaw_roll - 0.1, 0, 50)
        return torch.exp(-yaw_roll * 100) - 0.01 * torch.norm(joint_diff, dim=1)

    def check_termination(self):
        """Check if environments need to be reset"""
        # we don't keep track of previous states, so we need to do this in a different way
        """
        t = len(self.reset_buf)
        self.height_history[:, t] = self.root_states[:, 2]
        self.window_size = 10

        above_threshold = self.episode_length_buf > self.window_size:

            self.reset_buf = torch.any(
                torch.abs(
                    torch.median(self.height_history[:, t - self.window_size : t]) - self.root_states[:, 2]
                )
                > 0.01,
                dim=1,
            )
        """
        self.time_out_buf = self.episode_length_buf > self.max_episode_length  # no terminal reward for time-outs
        self.reset_buf = self.time_out_buf

    def _reward_base_height(self):
        """Calculates the reward based on the robot's base height. Penalizes deviation from a target base height.
        The reward is computed based on the height difference between the robot's base and the average height
        of its feet when they are in contact with the ground.
        """
        base_height = self.root_states[:, 2] - default_feet_height

        return base_height

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
