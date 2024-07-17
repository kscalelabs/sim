"""This module contains the configuration dataclasses for the robot."""

from dataclasses import dataclass, field

import numpy as np


# Default velocity command
@dataclass
class cmd:
    vx: float = 0.4
    vy: float = 0.0
    dyaw: float = 0.0


@dataclass
class Normalization:
    """Normalization constants for observations and actions."""

    @dataclass
    class obs_scales:
        lin_vel: float = 2.0
        ang_vel: float = 1.0
        dof_pos: float = 1.0
        dof_vel: float = 0.05
        quat: float = 1.0
        height_measurements: float = 5.0

    clip_observations: float = 18.0
    clip_actions: float = 18.0


@dataclass
class RobotConfig:
    """This dataclass contains various parameters and settings for configuring the robot.

    Attributes:
        dof: The number of degrees of freedom of the robot.
        num_actions: The number of actions the robot can perform.
        kps: The proportional gains for the PD controller.
        kds: The derivative gains for the PD controller.
        tau_limit: The torque limits for the robot's joints.
        robot_model_path: The path to the robot model file.
        dt: The simulation time step.
        phase_duration: The duration of each phase in the robot's gait.
        duration: The total duration of the simulation.
        decimation: The decimation factor for updating the policy.
        frame_stack: The number of frames to stack for observation.
        num_single_obs: The number of single observations.
        action_scale: The scaling factor for actions.
        num_observations: The total number of observations (frame_stack * num_single_obs).
        normalization: The normalization constants for observations and actions.
    """

    dof: int
    kps: np.ndarray
    kds: np.ndarray
    tau_limit: np.ndarray
    robot_model_path: str
    # is2ac
    dt: float = 0.00002  # 0.001
    phase_duration: float = 0.64
    duration: float = 10.0
    decimation: int = 10
    frame_stack: int = 15
    num_single_obs: int = 47
    action_scale: float = 0.25

    normalization: Normalization = field(default_factory=Normalization)

    @property
    def num_actions(self) -> int:
        return self.dof

    @property
    def num_observations(self) -> int:
        return self.frame_stack * self.num_single_obs
