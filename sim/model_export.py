"""Script to convert weights to Rust-compatible format."""

import re
from dataclasses import dataclass, fields
from io import BytesIO
from typing import Dict, List, Optional, Tuple

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
        vector_command: Tensor,  # (x_vel, y_vel, rot)
        timestamp: Tensor,  # current policy time (sec)
        dof_pos: Tensor,  # current angular position of the DoFs relative to default
        dof_vel: Tensor,  # current angular velocity of the DoFs
        prev_actions: Tensor,  # previous actions taken by the model
        imu_ang_vel: Tensor,  # angular velocity of the IMU
        imu_euler_xyz: Tensor,  # euler angles of the IMU
        hist_obs: Tensor,  # buffer of previous observations
    ) -> Dict[str, Tensor]:
        """Runs the actor model forward pass.

        Args:
            vector_command: The target velocity vector, with shape (3). It consistes of x_vel, y_vel, and rot.
            timestamp: The current policy time step, with shape (1).
            dof_pos: The current angular position of the DoFs relative to default, with shape (num_actions).
            dof_vel: The current angular velocity of the DoFs, with shape (num_actions).
            prev_actions: The previous actions taken by the model, with shape (num_actions).
            imu_ang_vel: The angular velocity of the IMU, with shape (3),
                in radians per second. If IMU is not used, can be all zeros.
            imu_euler_xyz: The euler angles of the IMU, with shape (3),
                in radians. "XYZ" means (roll, pitch, yaw). If IMU is not used,
                can be all zeros.
            state_tensor: The buffer of previous actions, with shape (frame_stack * num_single_obs). This is
                the return value of the previous forward pass. On the first
                pass, it should be all zeros.

        Returns:
            actions_scaled: The actions to take, with shape (num_actions), scaled by the action_scale.
            actions: The actions to take, with shape (num_actions).
            x: The new buffer of observations, with shape (frame_stack * num_single_obs).
        """
        sin_pos = torch.sin(2 * torch.pi * timestamp / self.cycle_time)
        cos_pos = torch.cos(2 * torch.pi * timestamp / self.cycle_time)

        x_vel, y_vel, rot = vector_command.split(1)

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
        q = dof_pos * self.dof_pos_scale
        dq = dof_vel * self.dof_vel_scale

        # Construct new observation
        new_x = torch.cat(
            (
                command_input,
                q,
                dq,
                prev_actions,
                imu_ang_vel.squeeze(0) * self.ang_vel_scale,
                imu_euler_xyz.squeeze(0) * self.quat_scale,
            ),
            dim=0,
        )

        # Clip the inputs
        new_x = torch.clamp(new_x, -self.clip_observations, self.clip_observations)

        # Add the new frame to the buffer
        x = torch.cat((hist_obs, new_x), dim=0)
        # Pop the oldest frame
        x = x[self.num_single_obs :]

        policy_input = x.unsqueeze(0)

        # Get actions from the policy
        actions = self.policy(policy_input).squeeze(0)
        actions_scaled = actions * self.action_scale

        return {"actions": actions_scaled, "actions_raw": actions, "new_x": x}


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

    # Add sim2sim metadata
    robot = a_model.robot
    robot_effort = robot.effort_mapping()
    robot_stiffness = robot.stiffness_mapping()
    robot_damping = robot.damping_mapping()
    num_actions = a_model.num_actions
    num_observations = a_model.num_observations

    default_standing = robot.default_standing()

    metadata = {
        "num_actions": num_actions,
        "num_observations": num_observations,
        "robot_effort": robot_effort,
        "robot_stiffness": robot_stiffness,
        "robot_damping": robot_damping,
        "default_standing": default_standing,
        "sim_dt": cfg.sim_dt,
        "sim_decimation": cfg.sim_decimation,
        "tau_factor": cfg.tau_factor,
        "action_scale": cfg.action_scale,
        "lin_vel_scale": cfg.lin_vel_scale,
        "ang_vel_scale": cfg.ang_vel_scale,
        "quat_scale": cfg.quat_scale,
        "dof_pos_scale": cfg.dof_pos_scale,
        "dof_vel_scale": cfg.dof_vel_scale,
        "frame_stack": cfg.frame_stack,
        "clip_observations": cfg.clip_observations,
        "clip_actions": cfg.clip_actions,
        "joint_names": robot.joint_names(),
    }

    return (
        a_model,
        metadata,
        input_tensors,
    )


def convert_model_to_onnx(model_path: str, cfg: ActorCfg, save_path: Optional[str] = None) -> ort.InferenceSession:
    """Converts a PyTorch model to a ONNX format.

    Args:
        model_path: Path to the PyTorch model.
        cfg: The configuration for the actor.
        save_path: Path to save the ONNX model.

    Returns:
        An ONNX inference session.
    """
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

    # Export the model to a buffer
    buffer = BytesIO()
    torch.onnx.export(jit_model, input_tensors, buffer)
    buffer.seek(0)

    # Load the model as an onnx model
    model_proto = onnx.load_model(buffer)

    # Add sim2sim metadata
    robot_effort = list(a_model.robot.effort().values())
    robot_stiffness = list(a_model.robot.stiffness().values())
    robot_damping = list(a_model.robot.damping().values())
    num_actions = a_model.num_actions
    num_observations = a_model.num_observations

    for field_name, field in [
        ("robot_effort", robot_effort),
        ("robot_stiffness", robot_stiffness),
        ("robot_damping", robot_damping),
        ("num_actions", num_actions),
        ("num_observations", num_observations),
    ]:
        meta = model_proto.metadata_props.add()
        meta.key = field_name
        meta.value = str(field)

    # Add the configuration of the model
    for field in fields(cfg):
        value = getattr(cfg, field.name)
        meta = model_proto.metadata_props.add()
        meta.key = field.name
        meta.value = str(value)

    if save_path:
        onnx.save_model(model_proto, save_path)

    # Convert model to bytes
    buffer2 = BytesIO()
    onnx.save_model(model_proto, buffer2)
    buffer2.seek(0)

    return ort.InferenceSession(buffer2.read())
