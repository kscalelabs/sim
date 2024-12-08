from typing import Tuple
from sim.model_export import ActorCfg, Actor
import torch


def get_actor_policy(model_path: str, cfg: ActorCfg) -> Tuple[torch.nn.Module, dict, Tuple[torch.Tensor, ...]]:
    model = torch.jit.load(model_path, map_location="cpu")
    a_model = Actor(model, cfg)

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
    robot_effort = list(a_model.robot.effort().values())
    robot_stiffness = list(a_model.robot.stiffness().values())
    robot_damping = list(a_model.robot.damping().values())
    num_actions = a_model.num_actions
    num_observations = a_model.num_observations

    return (
        a_model,
        {
            "robot_effort": robot_effort,
            "robot_stiffness": robot_stiffness,
            "robot_damping": robot_damping,
            "num_actions": num_actions,
            "num_observations": num_observations,
        },
        input_tensors,
    )
