"""Policy module."""

import math
from abc import ABC, abstractmethod
from collections import deque

from pathlib import Path
from typing import Union

import numpy as np
import torch

import utils
from sim.deploy.config import cmd, RobotConfig


class Policy(ABC):
    """Abstract base class for policies.

    Attributes:
        cmd: Command parameters.
        hist_obs: History of observations.
        model: Loaded PyTorch model.
        action: Current action.
    """
    cmd: cmd

    def __init__(
        self,
        model_path: Union[str, Path],
        obs_size: int,
        frame_stack: int = 3,
        num_actions: int = 12,
    ) -> None:
        self.hist_obs = deque()
        for _ in range(frame_stack):
            self.hist_obs.append(np.zeros([1, obs_size], dtype=np.double))
        self.model = torch.jit.load(model_path)
        self.action = np.zeros((num_actions), dtype=np.double)

    @abstractmethod
    def next_action(self, obs: np.ndarray) -> np.ndarray:
        """Computes the next action based on the observation.

        Args:
            obs: Observation.

        Returns:
            Next action.
        """
        pass

    @abstractmethod
    def pd_control(self, target_dof_pos: np.ndarray, dof_pos: np.ndarray, kp: float, dof_vel: np.ndarray, kd: float) -> np.ndarray:
        """Calculates torques from position commands using PD control.

        Args:
            target_dof_pos: Target generalized positions.
            dof_pos: Current generalized positions.
            kp: Proportional gain.
            dof_vel: Current generalized velocities.
            kd: Derivative gain.

        Returns:
            Calculated torques.
        """
        pass


class SimPolicy(Policy):
    """Policy for simulation.

    Attributes:
        dof: Degrees of freedom.
        cfg: Robot configuration.
    """

    def __init__(self, model_path: Union[str, Path], cfg: RobotConfig) -> None:
        self.cfg = cfg
        super().__init__(model_path, cfg.num_single_obs, cfg.frame_stack, cfg.num_actions)

    def pd_control(
        self,
        target_dof_pos: np.ndarray,
        dof_pos: np.ndarray,
        kp: float,
        dof_vel: np.ndarray,
        kd: float,
    ) -> np.ndarray:
        """
        Calculates torques from position commands using PD control.

        Args:
            target_dof_pos: Target generalized positions.
            dof_pos: Current generalized positions.
            kp: Proportional gain.
            target_dof_vel: Target generalized velocities.
            dof_vel: Current generalized velocities.
            kd: Derivative gain.

        Returns:
            Calculated torques.
        """
        target_dof_vel = np.zeros(self.cfg.num_actions, dtype=np.double)
        return (target_dof_pos - dof_pos) * kp + (target_dof_vel - dof_vel) * kd


    def parse_action(
        self, dof_pos: np.ndarray, dof_vel: np.ndarray, eu_ang: np.ndarray, ang_vel: np.ndarray, count_lowlevel: int
    ) -> np.ndarray:
        """Parses the action from the observation.
        Args:
            dof_pos: Joint positions.
            dof_vel: Joint velocities.
            eu_ang: Euler angles.
            ang_vel: Angular velocity.
            count_lowlevel: Low-level count.

        Returns:
            Parsed action.
        """
        obs = np.zeros([1, self.cfg.num_single_obs], dtype=np.float32)
        obs[0, 0] = math.sin(2 * math.pi * count_lowlevel * self.cfg.dt / self.cfg.phase_duration)
        obs[0, 1] = math.cos(2 * math.pi * count_lowlevel * self.cfg.dt / self.cfg.phase_duration)
        obs[0, 2] = cmd.vx * self.cfg.normalization.obs_scales.lin_vel
        obs[0, 3] = cmd.vy * self.cfg.normalization.obs_scales.lin_vel
        obs[0, 4] = cmd.dyaw * self.cfg.normalization.obs_scales.ang_vel
        obs[0, 5 : (self.cfg.num_actions + 5)] = dof_pos * self.cfg.normalization.obs_scales.dof_pos
        obs[0, (self.cfg.num_actions + 5) : (2 * self.cfg.num_actions + 5)] = (
            dof_vel * self.cfg.normalization.obs_scales.dof_vel
        )
        obs[0, (2 * self.cfg.num_actions + 5) : (3 * self.cfg.num_actions + 5)] = self.action
        obs[0, (3 * self.cfg.num_actions + 5) : (3 * self.cfg.num_actions + 5) + 3] = ang_vel
        obs[0, (3 * self.cfg.num_actions + 5) + 3 : (3 * self.cfg.num_actions + 5) + 2 * 3] = eu_ang
        obs = np.clip(obs, -self.cfg.normalization.clip_observations, self.cfg.normalization.clip_observations)
        return obs

    def next_action(
        self,
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
        orientation: np.ndarray,
        ang_vel: np.ndarray,
        count_lowlevel: int,
    ) -> np.ndarray:
        """Computes the next action based on the observation.

        Args:
            dof_pos: Joint positions.
            dof_vel: Joint velocities.
            orientation: Orientation quaternion.
            ang_vel: Angular velocity.
            count_lowlevel: Low-level count.

        Returns:
            Next action.
        """
        eu_ang = utils.quaternion_to_euler_array(orientation)
        eu_ang[eu_ang > math.pi] -= 2 * math.pi

        obs = self.parse_action(dof_pos, dof_vel, eu_ang, ang_vel, count_lowlevel)
        self.hist_obs.append(obs)
        self.hist_obs.popleft()
        policy_input = np.zeros([1, self.cfg.num_observations], dtype=np.float32)

        for i in range(self.cfg.frame_stack):
            policy_input[0, i * self.cfg.num_single_obs : (i + 1) * self.cfg.num_single_obs] = self.hist_obs[i][0, :]  # noqa

        self.action[:] = self.model(torch.tensor(policy_input))[0].detach().numpy()

        self.action = np.clip(self.action, -self.cfg.normalization.clip_actions, self.cfg.normalization.clip_actions)
        return self.action


class RealPolicy(Policy):
    """Policy for real-world deployment."""
    pass