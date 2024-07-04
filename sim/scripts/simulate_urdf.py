# mypy: disable-error-code="valid-newtype"
"""Defines a simple demo script for dropping a URDF to observe the physics.

This script demonstrates some good default physics parameters for the
simulation which avoid some of the "blowing up" issues that can occur with
default parameters. It also demonstrates how to load a URDF and drop it into
the simulation, and how to configure the DOF properties of the robot.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, Literal, NewType

from isaacgym import gymapi, gymtorch, gymutil
from sim.env import stompy_urdf_path
from sim.logging import configure_logging
from sim.stompy2.joints import Stompy

logger = logging.getLogger(__name__)

Gym = NewType("Gym", Any)
Env = NewType("Env", Any)
Sim = NewType("Sim", Any)
Robot = NewType("Robot", Any)
Viewer = NewType("Viewer", Any)
Args = NewType("Args", Any)

# Importing torch down here to avoid gymtorch issues.
import torch  # noqa: E402

# DRIVE_MODE = gymapi.DOF_MODE_EFFORT
DRIVE_MODE = gymapi.DOF_MODE_POS

# Stiffness and damping are Kp and Kd parameters for the PD controller
# that drives the joints to the desired position.
STIFFNESS = 80.0
DAMPING = 5.0
# STIFFNESS = 0.0
# DAMPING = 0.0

# Armature is a parameter that can be used to model the inertia of the joint.
# We set it to zero because the URDF already models the inertia of the joints.
ARMATURE = 0.0


@dataclass
class GymParams:
    gym: Gym
    env: Env
    sim: Sim
    robot: Robot
    viewer: Viewer
    args: Args


def load_gym() -> GymParams:
    # Initialize gym.
    gym = gymapi.acquire_gym()

    # Parse arguments.
    args = gymutil.parse_arguments(description="Joint control methods")

    # Sets the simulation parameters.
    sim_params = gymapi.SimParams()
    sim_params.substeps = 1
    sim_params.dt = 0.005

    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.contact_offset = -0.01
    sim_params.physx.rest_offset = -0.015
    sim_params.physx.bounce_threshold_velocity = 0.5
    sim_params.physx.max_depenetration_velocity = 10.0
    sim_params.physx.max_gpu_contact_pairs = 2**24
    sim_params.physx.default_buffer_size_multiplier = 5
    sim_params.physx.contact_collection = gymapi.CC_ALL_SUBSTEPS

    sim_params.physx.num_threads = args.num_threads
    sim_params.physx.use_gpu = args.use_gpu

    # sim_params.use_gpu_pipeline = False
    # if args.use_gpu_pipeline:
    #     warnings.warn("Forcing CPU pipeline.")

    # Creates the simulation.
    sim = gym.create_sim(
        args.compute_device_id,
        args.graphics_device_id,
        args.physics_engine,
        sim_params,
    )
    if sim is None:
        raise RuntimeError("Failed to create sim")

    # Creates a viewer.
    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    if viewer is None:
        raise RuntimeError("Failed to create viewer")

    # Add ground plane.
    plane_params = gymapi.PlaneParams()
    plane_params.restitution = 1.0
    gym.add_ground(sim, plane_params)

    # Set up the environment grid.
    num_envs = 1
    spacing = 1.5
    env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
    env_upper = gymapi.Vec3(spacing, 0.0, spacing)

    env = gym.create_env(sim, env_lower, env_upper, num_envs)

    # Loads the robot asset.
    asset_options = gymapi.AssetOptions()
    asset_options.default_dof_drive_mode = DRIVE_MODE
    asset_options.collapse_fixed_joints = True
    asset_options.disable_gravity = True
    asset_options.fix_base_link = True
    asset_path = stompy_urdf_path()
    robot_asset = gym.load_urdf(sim, str(asset_path.parent), str(asset_path.name), asset_options)

    # Adds the robot to the environment.
    initial_pose = gymapi.Transform()
    initial_pose.p = gymapi.Vec3(0.0, 5.0, 0.0)
    initial_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
    robot = gym.create_actor(env, robot_asset, initial_pose, "robot")

    # Configure DOF properties.
    props = gym.get_actor_dof_properties(env, robot)
    props["driveMode"] = DRIVE_MODE
    props["stiffness"].fill(STIFFNESS)
    props["damping"].fill(DAMPING)
    props["armature"].fill(ARMATURE)
    gym.set_actor_dof_properties(env, robot, props)

    # Look at the first environment.
    cam_pos = gymapi.Vec3(8, 4, 1.5)
    cam_target = gymapi.Vec3(0, 2, 1.5)
    gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

    # Gets tensors for the DOF states.
    dof_state_tensor = gym.acquire_dof_state_tensor(sim)
    gym.refresh_dof_state_tensor(sim)
    dof_state = gymtorch.wrap_tensor(dof_state_tensor)
    num_dof = len(Stompy.all_joints())
    dof_pos = dof_state.view(1, num_dof, 2)[..., 0]
    # dof_vel = dof_state.view(1, num_dof, 2)[..., 1]

    # Resets the DOF positions to the starting positions.
    # dof_vel[:] = 0.0
    starting_positions = Stompy.default_standing()
    dof_ids: Dict[str, int] = gym.get_actor_dof_dict(env, robot)
    print(starting_positions)
    for joint_name, joint_position in starting_positions.items():
        dof_pos[0, dof_ids[joint_name]] = joint_position
    env_ids_int32 = torch.zeros(1, dtype=torch.int32)
    gym.set_dof_state_tensor_indexed(
        sim,
        gymtorch.unwrap_tensor(dof_state),
        gymtorch.unwrap_tensor(env_ids_int32),
        1,
    )
    print(dof_pos)

    return GymParams(
        gym=gym,
        env=env,
        sim=sim,
        robot=robot,
        viewer=viewer,
        args=args,
    )


def run_gym(gym: GymParams, mode: Literal["one_at_a_time", "all_at_once"] = "all_at_once") -> None:
    # joints = Stompy.all_joints()
    # last_time = time.time()

    # dof_ids: Dict[str, int] = gym.gym.get_actor_dof_dict(gym.env, gym.robot)
    # body_ids: List[str] = gym.gym.get_actor_rigid_body_names(gym.env, gym.robot)

    # joint_id = 0
    # effort = 5.0

    while not gym.gym.query_viewer_has_closed(gym.viewer):
        gym.gym.simulate(gym.sim)
        gym.gym.fetch_results(gym.sim, True)
        gym.gym.step_graphics(gym.sim)
        gym.gym.draw_viewer(gym.viewer, gym.sim, True)
        gym.gym.sync_frame_time(gym.sim)

        # Print the joint forces.
        # print(gym.gym.get_actor_dof_forces(gym.env, gym.robot))
        # print(gym.gym.get_env_rigid_contact_forces(gym.env))

        # Prints the contact forces.
        # net_contact_forces = gym.gym.acquire_net_contact_force_tensor(gym.sim)
        # net_contact_forces_tensor = gymtorch.wrap_tensor(net_contact_forces).norm(2, dim=-1)  # (38, 3)
        # if net_contact_forces_tensor.size(0) != len(body_ids):
        #     raise ValueError("Mismatch between body IDs and contact forces.")
        # for body_id, contact_force in zip(body_ids, net_contact_forces_tensor):
        #     contact_force = contact_force.item()
        #     logger.info("Body %s: %.3g", body_id, contact_force)

        # Prints the joint angles.
        # joint_positions = gym.gym.get_actor_dof_states(gym.env, gym.robot, gymapi.STATE_ALL)
        # for joint_name, (joint_position, joint_velocity) in zip(joints, joint_positions):
        #     logger.info("Joint %s: %.3g %.3g", joint_name, joint_position, joint_velocity)

        # Every second, set the target effort for each joint to the reverse.
        # curr_time = time.time()

        # if mode == "one_at_a_time":
        #     if curr_time - last_time > 0.25:
        #         last_time = curr_time
        #         gym.gym.apply_dof_effort(gym.env, joint_id, 0.0)
        #         joint_id += 1
        #         if joint_id >= len(joints):
        #             effort = -effort
        #             joint_id = 0
        #         gym.gym.apply_dof_effort(gym.env, joint_id, effort)

        # elif mode == "all_at_once":
        #     if curr_time - last_time > 1.0:
        #         last_time = curr_time
        #         effort = -effort
        #         for joint_name in joints:
        #             gym.gym.apply_dof_effort(gym.env, dof_ids[joint_name], effort)

        # else:
        #     raise ValueError(f"Invalid mode: {mode}")

    gym.gym.destroy_viewer(gym.viewer)
    gym.gym.destroy_sim(gym.sim)


def main() -> None:
    configure_logging()
    gym = load_gym()
    run_gym(gym)


if __name__ == "__main__":
    # python -m sim.scripts.drop_urdf
    main()
