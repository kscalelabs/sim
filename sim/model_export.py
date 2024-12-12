"""Script to convert weights to Rust-compatible format."""

import re
from dataclasses import dataclass, fields
from io import BytesIO
from typing import List, Optional, Tuple

import onnx
import onnxruntime as ort
import torch
from scripts.create_fixed_torso import load_embodiment
from torch import Tensor, nn
from torch.distributions import Normal


@dataclass
class ActorCfg:
    embodiment: str
    cycle_time: float  # Cycle time for sinusoidal command input
    action_scale: float  # Scale for actions
    lin_vel_scale: float  # Scale for linear velocity
    ang_vel_scale: float  # Scale for angular velocity
    quat_scale: float  # Scale for quaternion
    dof_pos_scale: float  # Scale for joint positions
    dof_vel_scale: float  # Scale for joint velocities
    frame_stack: int  # Number of frames to stack for the policy input
    clip_observations: float  # Clip observations to this value
    clip_actions: float  # Clip actions to this value
    sim_dt: float  # Simulation time step
    sim_decimation: int  # Simulation decimation
    tau_factor: float  # Torque limit factor


class ActorCritic(nn.Module):
    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        actor_hidden_dims: List[int] = [256, 256, 256],
        critic_hidden_dims: List[int] = [256, 256, 256],
        init_noise_std: float = 1.0,
        activation: nn.Module = nn.ELU(),
    ) -> None:
        super(ActorCritic, self).__init__()

        mlp_input_dim_a = num_actor_obs
        mlp_input_dim_c = num_critic_obs

        # Policy function.
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)  # type: ignore[arg-type]
        for dim_i in range(len(actor_hidden_dims)):
            if dim_i == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[dim_i], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[dim_i], actor_hidden_dims[dim_i + 1]))
                actor_layers.append(activation)  # type: ignore[arg-type]
        self.actor = nn.Sequential(*actor_layers)

        # Value function.
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)  # type: ignore[arg-type]
        for dim_i in range(len(critic_hidden_dims)):
            if dim_i == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[dim_i], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[dim_i], critic_hidden_dims[dim_i + 1]))
                critic_layers.append(activation)  # type: ignore[arg-type]
        self.critic = nn.Sequential(*critic_layers)

        # Action noise.
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None

        # Disable args validation for speedup.
        Normal.set_default_validate_args = False  # type: ignore[unused-ignore,assignment,method-assign,attr-defined]


class Actor(nn.Module):
    """Actor model.

    Parameters:
        policy: The policy network.
        cfg: The configuration for the actor.
    """

    def __init__(self, policy: nn.Module, cfg: ActorCfg) -> None:
        super().__init__()

        self.robot = load_embodiment(cfg.embodiment)

        # Policy config
        default_dof_pos_dict = self.robot.default_standing()
        self.num_actions = len(self.robot.all_joints())
        self.frame_stack = cfg.frame_stack

        # 11 is the number of single observation features - 6 from IMU, 5 from command input
        # 3 comes from the number of times num_actions is repeated in the observation (dof_pos, dof_vel, prev_actions)
        self.num_single_obs = 11 + self.num_actions * 3
        self.num_observations = int(self.frame_stack * self.num_single_obs)

        self.policy = policy

        # This is the policy reference joint positions and should be the same order as the policy and mjcf file.
        # CURRENTLY NOT USED IN FORWARD PASS TO MAKE MORE GENERALIZEABLE FOR REAL AND SIM
        self.default_dof_pos = torch.tensor(list(default_dof_pos_dict.values()))

        self.action_scale = cfg.action_scale
        self.lin_vel_scale = cfg.lin_vel_scale
        self.ang_vel_scale = cfg.ang_vel_scale
        self.quat_scale = cfg.quat_scale
        self.dof_pos_scale = cfg.dof_pos_scale
        self.dof_vel_scale = cfg.dof_vel_scale

        self.clip_observations = cfg.clip_observations
        self.clip_actions = cfg.clip_actions

        self.cycle_time = cfg.cycle_time

    def get_init_buffer(self) -> Tensor:
        return torch.zeros(self.num_observations)

    def forward(
        self,
        command_input: Tensor,  # x-coordinate of the target velocity
        t: Tensor,  # current policy time (sec)
        joint_positions: Tensor,  # current angular position of the DoFs relative to default
        joint_velocities: Tensor,  # current angular velocity of the DoFs
        prev_actions: Tensor,  # previous actions taken by the model
        angular_velocities: Tensor,  # angular velocity of the IMU
        euler_rotation: Tensor,  # euler angles of the IMU
        buffer: Tensor,  # buffer of previous observations
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Runs the actor model forward pass.

        Args:
            command_input: The command input, with shape (3).
            t: The current policy time step, with shape (1).
            joint_positions: The current angular position of the DoFs relative to default, with shape (num_actions).
            joint_velocities: The current angular velocity of the DoFs, with shape (num_actions).
            prev_actions: The previous actions taken by the model, with shape (num_actions).
            angular_velocities: The angular velocity of the IMU, with shape (3),
                in radians per second. If IMU is not used, can be all zeros.
            euler_rotation: The euler angles of the IMU, with shape (3),
                in radians. "XYZ" means (roll, pitch, yaw). If IMU is not used,
                can be all zeros.
            buffer: The buffer of previous actions, with shape (frame_stack * num_single_obs). This is
                the return value of the previous forward pass. On the first
                pass, it should be all zeros.

        Returns:
            actions_scaled: The actions to take, with shape (num_actions), scaled by the action_scale.
            actions: The actions to take, with shape (num_actions).
            x: The new buffer of observations, with shape (frame_stack * num_single_obs).
        """
        sin_pos = torch.sin(2 * torch.pi * t / self.cycle_time)
        cos_pos = torch.cos(2 * torch.pi * t / self.cycle_time)

        # Construct command input
        command_input = torch.cat(
            (
                sin_pos,
                cos_pos,
                command_input[0] * self.lin_vel_scale,
                command_input[1] * self.lin_vel_scale,
                command_input[2] * self.ang_vel_scale,
            ),
            dim=0,
        )

        # Calculate current position and velocity observations
        q = joint_positions * self.dof_pos_scale
        dq = joint_velocities * self.dof_vel_scale

        # Construct new observation
        new_x = torch.cat(
            (
                command_input,
                q,
                dq,
                prev_actions,
                angular_velocities * self.ang_vel_scale,
                euler_rotation * self.quat_scale,
            ),
            dim=0,
        )

        # Clip the inputs
        new_x = torch.clamp(new_x, -self.clip_observations, self.clip_observations)

        # Add the new frame to the buffer
        x = torch.cat((buffer, new_x), dim=0)
        # Pop the oldest frame
        x = x[self.num_single_obs :]

        policy_input = x.unsqueeze(0)

        # Get actions from the policy
        actions = self.policy(policy_input).squeeze(0)
        actions_scaled = actions * self.action_scale

        return actions_scaled, actions, x


def get_actor_policy(model_path: str, cfg: ActorCfg) -> Tuple[nn.Module, dict, Tuple[Tensor, ...]]:
    all_weights = torch.load(model_path, map_location="cpu", weights_only=True)
    weights = all_weights["model_state_dict"]
    num_actor_obs = weights["actor.0.weight"].shape[1]
    num_critic_obs = weights["critic.0.weight"].shape[1]
    num_actions = weights["std"].shape[0]
    actor_hidden_dims = [v.shape[0] for k, v in weights.items() if re.match(r"actor\.\d+\.weight", k)]
    critic_hidden_dims = [v.shape[0] for k, v in weights.items() if re.match(r"critic\.\d+\.weight", k)]
    actor_hidden_dims = actor_hidden_dims[:-1]
    critic_hidden_dims = critic_hidden_dims[:-1]

    ac_model = ActorCritic(num_actor_obs, num_critic_obs, num_actions, actor_hidden_dims, critic_hidden_dims)
    ac_model.load_state_dict(weights)

    a_model = Actor(ac_model.actor, cfg)

    # Gets the model input tensors.
    command_input = torch.randn(3)
    t = torch.randn(1)
    joint_positions = torch.randn(a_model.num_actions)
    joint_velocities = torch.randn(a_model.num_actions)
    prev_actions = torch.randn(a_model.num_actions)
    angular_velocities = torch.randn(3)
    euler_rotation = torch.randn(3)
    buffer = a_model.get_init_buffer()
    input_tensors = (
        command_input,
        t,
        joint_positions,
        joint_velocities,
        prev_actions,
        angular_velocities,
        euler_rotation,
        buffer,
    )

    # Add sim2sim metadata
    robot_effort = list(a_model.robot.effort().values())
    robot_stiffness = list(a_model.robot.stiffness().values())
    robot_damping = list(a_model.robot.damping().values())
    num_actions = a_model.num_actions
    num_observations = a_model.num_observations

    default_standing = list(a_model.robot.default_standing().values())

    return (
        a_model,
        {
            "robot_effort": robot_effort,
            "robot_stiffness": robot_stiffness,
            "robot_damping": robot_damping,
            "num_actions": num_actions,
            "num_observations": num_observations,
            "default_standing": default_standing,
        },
        input_tensors,
    )
