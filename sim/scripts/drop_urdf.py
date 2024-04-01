"""Defines a simple demo script for dropping a URDF to observe the physics.

This script demonstrates some good default physics parameters for the
simulation which avoid some of the "blowing up" issues that can occur with
default parameters. It also demonstrates how to load a URDF and drop it into
the simulation, and how to configure the DOF properties of the robot.
"""

import logging
import time
from dataclasses import dataclass
from typing import Any, Literal

from isaacgym import gymapi, gymutil

from sim.env import stompy_urdf_path
from sim.logging import configure_logging
from sim.stompy.joints import Stompy

logger = logging.getLogger(__name__)

Gym = Any
Env = Any
Sim = Any
Robot = Any
Viewer = Any
Args = Any

DRIVE_MODE = gymapi.DOF_MODE_EFFORT
# DRIVE_MODE = gymapi.DOF_MODE_POS

# Stiffness and damping are Kp and Kd parameters for the PD controller
# that drives the joints to the desired position.
# STIFFNESS = 80.0
# DAMPING = 5.0
STIFFNESS = 0.0
DAMPING = 0.0

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
    sim_params.physx.num_velocity_iterations = 0
    sim_params.physx.contact_offset = 0.02
    sim_params.physx.rest_offset = 0.01
    sim_params.physx.bounce_threshold_velocity = 1.0
    sim_params.physx.max_depenetration_velocity = 0.5
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
    asset_path = stompy_urdf_path()
    robot_asset = gym.load_urdf(sim, str(asset_path.parent), str(asset_path.name), asset_options)

    # Adds the robot to the environment.
    initial_pose = gymapi.Transform()
    initial_pose.p = gymapi.Vec3(0.0, 5.0, 0.0)
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

    return GymParams(
        gym=gym,
        env=env,
        sim=sim,
        robot=robot,
        viewer=viewer,
        args=args,
    )


def run_gym(gym: GymParams, mode: Literal["one_at_a_time", "all_at_once"] = "all_at_once") -> None:
    joints = Stompy.all_joints()
    last_time = time.time()
    dof_ids = gym.gym.get_actor_dof_dict(gym.env, gym.robot)

    joint_id = 0
    effort = 10.0

    while not gym.gym.query_viewer_has_closed(gym.viewer):
        gym.gym.simulate(gym.sim)
        gym.gym.fetch_results(gym.sim, True)
        gym.gym.step_graphics(gym.sim)
        gym.gym.draw_viewer(gym.viewer, gym.sim, True)
        gym.gym.sync_frame_time(gym.sim)

        # Every second, set the target effort for each joint to the reverse.
        curr_time = time.time()

        if mode == "one_at_a_time":
            if curr_time - last_time > 0.25:
                last_time = curr_time
                gym.gym.apply_dof_effort(gym.env, joint_id, 0.0)
                joint_id += 1
                if joint_id >= len(joints):
                    effort = -effort
                    joint_id = 0
                gym.gym.apply_dof_effort(gym.env, joint_id, effort)

        elif mode == "all_at_once":
            if curr_time - last_time > 1.0:
                last_time = curr_time
                effort = -effort
                for joint_name in joints:
                    gym.gym.apply_dof_effort(gym.env, dof_ids[joint_name], effort)

        else:
            raise ValueError(f"Invalid mode: {mode}")

    gym.gym.destroy_viewer(gym.viewer)
    gym.gym.destroy_sim(gym.sim)


def main() -> None:
    configure_logging()
    gym = load_gym()
    run_gym(gym)


if __name__ == "__main__":
    # python -m sim.scripts.drop_urdf
    main()
