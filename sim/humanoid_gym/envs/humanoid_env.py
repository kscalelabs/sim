# mypy: disable-error-code="valid-newtype"
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

NUM_JOINTS = len(Stompy.all_joints())  # 35


class StompyFreeEnv(LeggedRobot):
    """StompyFreeEnv adapts the XBot environment for training Stompy.

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
        self.compute_observations()

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
        noise_vec[8:45] = noise_scales.dof_pos * self.obs_scales.dof_pos
        noise_vec[45:82] = noise_scales.dof_vel * self.obs_scales.dof_vel
        noise_vec[82:119] = 0.0  # Previous actions
        noise_vec[119:121] = noise_scales.quat * self.obs_scales.quat  # Gyroscope (x, y)
        return noise_vec

    def step(self, actions: Tensor) -> Tensor:
        # Dynamic randomization. We simulate a delay in the actions to
        # help the policy generalize better in the real world.
        delay = torch.rand((self.num_envs, 1), device=self.device)
        actions = (1 - delay) * actions + delay * self.actions
        actions += self.cfg.domain_rand.dynamic_randomization * torch.randn_like(actions) * actions
        return super().step(actions)

    def compute_observations(self) -> None:
        contact_mask = self.contact_forces[:, self.feet_indices, 2] > 5.0

        self.command_input = self.commands

        q = self.dof_pos * self.obs_scales.dof_pos
        dq = self.dof_vel * self.obs_scales.dof_vel

        self.privileged_obs_buf = torch.cat(
            (
                self.command_input,  # 8
                self.actions,  # 37
                q,  # 37
                dq,  # 37
                self.base_lin_vel * self.obs_scales.lin_vel,  # 3
                self.base_ang_vel * self.obs_scales.ang_vel,  # 3
                self.base_euler_xyz * self.obs_scales.quat,  # 3
                self.rand_push_force[:, :2],  # 2
                self.rand_push_torque,  # 3
                self.env_frictions,  # 1
                self.body_mass / 30.0,  # 1
                contact_mask,  # 2
            ),
            dim=-1,
        )

        self.obs_buf = torch.cat(
            (
                self.command_input,  # 8
                q,  # 37
                dq,  # 37
                self.actions,  # 37
                self.base_euler_xyz[:, (0, 2)] * self.obs_scales.quat,  # 2
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
            obs_now = (
                self.obs_buf.clone()
                + torch.randn_like(self.obs_buf) * self.noise_scale_vec * self.cfg.noise.noise_level
            )
        else:
            obs_now = self.obs_buf.clone()
        self.obs_history.append(obs_now)
        self.critic_history.append(self.privileged_obs_buf)

        obs_buf_all = torch.stack([self.obs_history[i] for i in range(self.obs_history.maxlen)], dim=1)  # N,T,K

        self.obs_buf = obs_buf_all.reshape(self.num_envs, -1)  # N, T*K
        self.privileged_obs_buf = torch.cat([self.critic_history[i] for i in range(self.cfg.env.c_frame_stack)], dim=1)

    def reset_idx(self, env_ids: Tensor) -> None:
        super().reset_idx(env_ids)

        for i in range(self.obs_history.maxlen):
            self.obs_history[i][env_ids] *= 0
        for i in range(self.critic_history.maxlen):
            self.critic_history[i][env_ids] *= 0

    def _reward_base_height(self) -> Tensor:
        """Calculates the reward based on the robot's base height.

        This rewards the robot for being close to the target height without
        going over (we actually penalise it for going over).

        Returns:
            The reward for maximizing the base height.
        """
        base_height = self.root_states[:, 2]
        reward = base_height / self.cfg.rewards.base_height_target
        reward[reward > 1.0] = 0.0
        return reward

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
