"""Script to convert weights to Rust-compatible format."""

import re
from dataclasses import dataclass
from io import BytesIO
from typing import List, Optional, Tuple

import numpy as np
import onnxruntime as ort
import torch
from scripts.create_mjcf import load_embodiment
from torch import Tensor, nn
from torch.distributions import Normal


@dataclass
class ActorCfg:
    embodiment: str = "stompypro"
    cycle_time: float = 0.4
    policy_dt: float = 1.0/50.0
    action_scale: float = 0.25
    lin_vel_scale: float = 2.0
    ang_vel_scale: float = 1.0
    quat_scale: float = 1.0
    dof_pos_scale: float = 1.0
    dof_vel_scale: float = 0.05
    frame_stack: int = 15
    clip_observations: float = 18.0
    clip_actions: float = 18.0


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
        actor_layers.append(activation)
        for dim_i in range(len(actor_hidden_dims)):
            if dim_i == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[dim_i], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[dim_i], actor_hidden_dims[dim_i + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # Value function.
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for dim_i in range(len(critic_hidden_dims)):
            if dim_i == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[dim_i], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[dim_i], critic_hidden_dims[dim_i + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        # Action noise.
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None

        # Disable args validation for speedup.
        Normal.set_default_validate_args = False


class Actor(nn.Module):
    """Actor model.

    Parameters:
        policy: The policy network.
        cycle_time: The cycle time of the model, in seconds.
        policy_dt: The time step of the policy, in seconds.
        action_scale: The scale of the actions.
        lin_vel_scale: The scale of the linear velocity.
        ang_vel_scale: The scale of the angular velocity.
        quat_scale: The scale of the quaternion.
        dof_pos_scale: The scale of the DoF positions.
        dof_vel_scale: The scale of the DoF velocities.
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
        self.policy_dt = cfg.policy_dt

    def get_init_buffer(self) -> Tensor:
        return torch.zeros(self.num_observations)

    def forward(
        self,
        x_vel: Tensor,
        y_vel: Tensor,
        rot: Tensor,
        t: Tensor,
        dof_pos: Tensor,
        dof_vel: Tensor,
        prev_actions: Tensor,
        imu_ang_vel: Tensor,
        imu_euler_xyz: Tensor,
        buffer: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, int]:
        """Runs the actor model forward pass.

        Args:
            x_vel: The x-coordinate of the target velocity, with shape (1).
            y_vel: The y-coordinate of the target velocity, with shape (1).
            rot: The target angular velocity, with shape (1).
            t: The current policy time step, with shape (1).
            dof_pos: The current angular position of the DoFs, with shape (num_actions).
            dof_vel: The current angular velocity of the DoFs, with shape (num_actions).
            prev_actions: The previous actions taken by the model, with shape (num_actions).
            imu_ang_vel: The angular velocity of the IMU, with shape (3),
                in radians per second. If IMU is not used, can be all zeros.
            imu_euler_xyz: The euler angles of the IMU, with shape (3),
                in radians. "XYZ" means (roll, pitch, yaw). If IMU is not used,
                can be all zeros.
            buffer: The buffer of previous actions, with shape (frame_stack * num_single_obs). This is
                the return value of the previous forward pass. On the first
                pass, it should be all zeros.

        Returns:
            The torques to apply to the DoFs, the actions taken, and the
            next buffer.
        """
        sin_pos = torch.sin(2 * torch.pi * t  / self.cycle_time)
        cos_pos = torch.cos(2 * torch.pi * t / self.cycle_time)

        # Construct command input
        command_input = torch.cat(
            (
                sin_pos,
                cos_pos,
                x_vel * self.lin_vel_scale,
                y_vel * self.lin_vel_scale,
                rot * self.ang_vel_scale,
            ),
            dim=0,
        )

        # Calculate current position and velocity observations
        q = (dof_pos - self.default_dof_pos) * self.dof_pos_scale
        dq = dof_vel * self.dof_vel_scale

        # Construct new observation
        new_x = torch.cat(
            (
                command_input,
                q,
                dq,
                prev_actions,
                imu_ang_vel * self.ang_vel_scale,
                imu_euler_xyz * self.quat_scale,
            ),
            dim=0,
        )

        # Clip the inputs
        new_x = torch.clamp(new_x, -self.clip_observations, self.clip_observations)

        # Add the new frame to the buffer and pop the oldest frame
        x = torch.cat((buffer, new_x), dim=0)
        x = x[self.num_single_obs:]

        policy_input = x.unsqueeze(0)
        # policy_input: np. = np.zeros([1, np.array(self.num_observations)], dtype=np.float32)
        # for i in range(self.frame_stack):
        #     start_index = i * self.num_single_obs
        #     end_index = (i + 1) * self.num_single_obs
        #     policy_input[0, start_index : end_index] = x[start_index : end_index]

        # Get actions from the policy
        actions = self.policy(policy_input).squeeze(0)
        actions_scaled = actions * self.action_scale


        return actions_scaled, actions, x, self.policy_dt


"""Converts a PyTorch model to a ONNX format."""
def convert(model_path: str, cfg: ActorCfg, save_path: Optional[str] = None) -> ort.InferenceSession:
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
    x_vel = torch.randn(1)
    y_vel = torch.randn(1)
    rot = torch.randn(1)
    t = torch.randn(1)
    dof_pos = torch.randn(a_model.num_actions)
    dof_vel = torch.randn(a_model.num_actions)
    prev_actions = torch.randn(a_model.num_actions)
    imu_ang_vel = torch.randn(3)
    imu_euler_xyz = torch.randn(3)
    buffer = a_model.get_init_buffer()
    input_tensors = (x_vel, y_vel, rot, t, dof_pos, dof_vel, prev_actions, imu_ang_vel, imu_euler_xyz, buffer)

    jit_model = torch.jit.script(a_model)

    if save_path:
        torch.onnx.export(jit_model, input_tensors, save_path)

    buffer = BytesIO()
    torch.onnx.export(jit_model, input_tensors, buffer)
    buffer.seek(0)

    return ort.InferenceSession(buffer.read())


if __name__ == "__main__":
    convert("position_control.pt", ActorCfg(), "position_control.onnx")