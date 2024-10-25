"""Script to convert weights to Rust-compatible format."""

import re

import torch
from torch import Tensor, nn
from torch.distributions import Normal

# TODO: wesley: these constants should be loaded by from the sim joints config
DOF_NAMES = [
    "L_hip_y",
    "L_hip_x",
    "L_hip_z",
    "L_knee",
    "L_ankle_y",
    "R_hip_y",
    "R_hip_x",
    "R_hip_z",
    "R_knee",
    "R_ankle_y",
]

BODY_NAMES = [
    "base",
    "trunk",
    "L_buttock",
    "L_leg",
    "L_thigh",
    "L_calf",
    "L_foot",
    "L_clav",
    "L_scapula",
    "L_uarm",
    "L_farm",
    "R_buttock",
    "R_leg",
    "R_thigh",
    "R_calf",
    "R_foot",
    "R_clav",
    "R_scapula",
    "R_uarm",
    "R_farm",
]

DEFAULT_JOINT_ANGLES = {
    "L_ankle_y": -0.258,
    "L_hip_y": -0.157,
    "L_hip_z": 0.0628,
    "L_hip_x": 0.0394,
    "L_knee": 0.441,
    "R_ankle_y": -0.223,
    "R_hip_y": -0.22,
    "R_hip_z": 0.0314,
    "R_hip_x": 0.026,
    "R_knee": 0.441,
}

STIFFNESS = {"hip_y": 120, "hip_x": 60, "hip_z": 60, "knee": 120, "ankle_y": 17}
DAMPING = {"hip_y": 10, "hip_x": 10, "hip_z": 10, "knee": 10, "ankle_y": 5}

NUM_ACTIONS = len(DOF_NAMES)


class Actor(nn.Module):
    def __init__(
        self, policy: nn.Module, 
        cycle_time: float = 0.4,
        action_scale: float = 0.25,
        lin_vel_scale: float = 2.0,
        ang_vel_scale: float = 1.0,
        quat_scale: float = 1.0,
        dof_pos_scale: float = 1.0,
        dof_vel_scale: float = 0.05,
    ) -> None:
        super().__init__()
        self.policy = policy
        self.default_dof_pos = torch.zeros(NUM_ACTIONS, dtype=torch.float)

        for i in range(len(DOF_NAMES)):
            name = DOF_NAMES[i]
            self.default_dof_pos[i] = DEFAULT_JOINT_ANGLES[name]

        self.action_scale = action_scale    
        self.lin_vel_scale = lin_vel_scale
        self.ang_vel_scale = ang_vel_scale
        self.quat_scale = quat_scale
        self.dof_pos_scale = dof_pos_scale
        self.dof_vel_scale = dof_vel_scale

        self.cycle_time = cycle_time

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
        t: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Runs the actor model forward pass.

        Args:
            x_vel: The x-coordinate of the target velocity, with shape (1).
            y_vel: The y-coordinate of the target velocity, with shape (1).
            rot: The target angular velocity, with shape (1).
            t: The current time, with shape (1).
            dof_pos: The current angular position of the DoFs, with shape (10).
            dof_vel: The current angular velocity of the DoFs, with shape (10).
            prev_actions: The previous actions taken by the model, with shape (10).
            imu_ang_vel: The angular velocity of the IMU, with shape (3),
                in radians per second. If IMU is not used, can be all zeros.
            imu_euler_xyz: The euler angles of the IMU, with shape (3),
                in radians. "XYZ" means (roll, pitch, yaw). If IMU is not used,
                can be all zeros.
            buffer: The buffer of previous actions, with shape (574). This is
                the return value of the previous forward pass. On the first
                pass, it should be all zeros.
            t: The current time, with shape (1).

        Returns:
            The torques to apply to the DoFs, the actions taken, and the
            next buffer.
        """
        sin_pos = torch.sin(2 * torch.pi * t / self.cycle_time)
        cos_pos = torch.cos(2 * torch.pi * t /self.cycle_time)

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

        q = (dof_pos - self.default_dof_pos) * self.dof_pos_scale
        dq = dof_vel * self.dof_vel_scale

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

        x = torch.cat((buffer, new_x), dim=0)

        actions = self.policy(x.unsqueeze(0)).squeeze(0)
        actions_scaled = actions * self.action_scale

        return actions_scaled, actions, x[41:]


def convert() -> None:
    all_weights = torch.load("position_control.pt", map_location="cpu", weights_only=True)
    weights = all_weights["model_state_dict"]
    num_actor_obs = weights["actor.0.weight"].shape[1]
    num_critic_obs = weights["critic.0.weight"].shape[1]
    num_actions = weights["std"].shape[0]

    actor_hidden_dims = [v.shape[0] for k, v in weights.items() if re.match(r"actor\.\d+\.weight", k)]
    critic_hidden_dims = [v.shape[0] for k, v in weights.items() if re.match(r"critic\.\d+\.weight", k)]
    actor_hidden_dims = actor_hidden_dims[:-1]
    critic_hidden_dims = critic_hidden_dims[:-1]

    # Wesley - this should be called from the sim model
    ac_model = ActorCritic(num_actor_obs, num_critic_obs, num_actions, actor_hidden_dims, critic_hidden_dims)
    ac_model.load_state_dict(weights)

    a_model = Actor(ac_model.actor)

    # Gets the model input tensors.
    x_vel = torch.randn(1)
    y_vel = torch.randn(1)
    rot = torch.randn(1)
    t = torch.randn(1)
    dof_pos = torch.randn(NUM_ACTIONS)
    dof_vel = torch.randn(NUM_ACTIONS)
    prev_actions = torch.randn(NUM_ACTIONS)
    imu_ang_vel = torch.randn(3)
    imu_euler_xyz = torch.randn(3)
    buffer = torch.zeros(574)
    input_tensors = (x_vel, y_vel, rot, t, dof_pos, dof_vel, prev_actions, imu_ang_vel, imu_euler_xyz, buffer)

    # Run the model once, for debugging.
    # a_model(*input_tensors)

    jit_model = torch.jit.script(a_model)
    torch.onnx.export(jit_model, input_tensors, "position_control.onnx")


if __name__ == "__main__":
    convert()
