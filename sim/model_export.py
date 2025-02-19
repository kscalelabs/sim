"""Script to convert weights to Rust-compatible format."""

import importlib
import re
from dataclasses import dataclass, fields
from io import BytesIO
from typing import Any, List, Optional, Tuple, Union

import onnx
import onnxruntime as ort
import torch
from torch import Tensor, nn
from torch.distributions import Normal


def load_embodiment(embodiment: str) -> Any:  # noqa: ANN401
    # Dynamically import embodiment
    module_name = f"sim.resources.{embodiment}.joints"
    module = importlib.import_module(module_name)
    robot = getattr(module, "Robot")
    return robot


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
    num_single_obs: int  # Number of single observation features
    clip_observations: float  # Clip observations to this value
    clip_actions: float  # Clip actions to this value
    num_single_obs: int  # Number of single observations
    num_actions: int  # Number of actions
    num_joints: int  # Number of joints
    sim_dt: float  # Simulation time step
    sim_decimation: int  # Simulation decimation
    pd_decimation: int  # PD decimation
    tau_factor: float  # Torque limit factor
    use_projected_gravity: bool  # Use projected gravity as IMU observation


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
        actor_layers: List[Union[nn.Linear, nn.Module]] = []
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
        critic_layers: List[Union[nn.Linear, nn.Module]] = []
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

        Normal.set_default_validate_args = False  # type: ignore[unused-ignore, assignment, method-assign]


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
        self.frame_stack = cfg.frame_stack

        self.num_actions = cfg.num_actions
        self.num_joints = cfg.num_joints

        # 11 is the number of single observation features - 6 from IMU, 5 from command input
        # 9 is the number of single observation features - 3 from IMU(quat), 5 from command input
        # 3 comes from the number of times num_actions is repeated in the observation (dof_pos, dof_vel, prev_actions)
        self.num_single_obs = cfg.num_single_obs  # 11 + self.num_actions * 3  # pfb30
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
        self.use_projected_gravity = cfg.use_projected_gravity

    def get_init_buffer(self) -> Tensor:
        return torch.zeros(self.num_observations)

    def forward(
        self,
        x_vel: Tensor,  # x-coordinate of the target velocity
        y_vel: Tensor,  # y-coordinate of the target velocity
        rot: Tensor,  # target angular velocity
        t: Tensor,  # current policy time (sec)
        dof_pos: Tensor,  # current angular position of the DoFs relative to default
        dof_vel: Tensor,  # current angular velocity of the DoFs
        prev_actions: Tensor,  # previous actions taken by the model
        projected_gravity: Tensor,  # quaternion of the IMU
        # imu_euler_xyz: Tensor,  # euler angles of the IMU
        # imu_ang_vel: Tensor,  # angular velocity of the IMU
        buffer: Tensor,  # buffer of previous observations
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Runs the actor model forward pass.

        Args:
            x_vel: The x-coordinate of the target velocity, with shape (1).
            y_vel: The y-coordinate of the target velocity, with shape (1).
            rot: The target angular velocity, with shape (1).
            t: The current policy time step, with shape (1).
            dof_pos: The current angular position of the DoFs relative to default, with shape (num_actions).
            dof_vel: The current angular velocity of the DoFs, with shape (num_actions).
            prev_actions: The previous actions taken by the model, with shape (num_actions).
            projected_gravity: The projected gravity vector, with shape (3), in meters per second squared.
            imu_ang_vel: The angular velocity of the IMU, with shape (3),
                in radians per second. If IMU is not used, can be all zeros.
            imu_euler_xyz: The euler angles of the IMU, with shape (3),
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
                projected_gravity,
                # imu_ang_vel,
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
    all_weights = torch.load(model_path, map_location="cpu")
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
    dof_pos = torch.randn(a_model.num_joints)
    dof_vel = torch.randn(a_model.num_joints)
    prev_actions = torch.randn(a_model.num_actions)
    projected_gravity = torch.randn(3)
    ang_vel = torch.randn(3)
    buffer = a_model.get_init_buffer()
    # input_tensors = (x_vel, y_vel, rot, t, dof_pos, dof_vel, prev_actions, projected_gravity, ang_vel, buffer)
    input_tensors = (x_vel,
                     y_vel,
                     rot,
                     t,
                     dof_pos,
                     dof_vel,
                     prev_actions,
                     projected_gravity,
                     buffer,
                     )

    # jit_model = torch.jit.script(a_model)

    # Add sim2sim metadata
    robot_effort = list(a_model.robot.effort().values())
    robot_stiffness = list(a_model.robot.stiffness().values())
    robot_damping = list(a_model.robot.damping().values())
    default_standing = list(a_model.robot.default_standing().values())
    num_actions = a_model.num_actions
    num_observations = a_model.num_observations

    return (
        a_model,
        {
            "robot_effort": robot_effort,
            "robot_stiffness": robot_stiffness,
            "robot_damping": robot_damping,
            "default_standing": default_standing,
            "num_actions": num_actions,
            "num_observations": num_observations,
            "num_joints": a_model.num_joints,
        },
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
    all_weights = torch.load(model_path, map_location="cpu")  # , weights_only=True)
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


if __name__ == "__main__":
    print("hi there :)")
