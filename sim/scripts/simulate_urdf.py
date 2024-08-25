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
import math
import time
from isaacgym import gymapi, gymtorch, gymutil
import numpy as np
from sim.env import robot_urdf_path
from sim.resources.stompymini.joints import Robot as Stompy
# Importing torch down here to avoid gymtorch issues.
import torch  # noqa: E402 #  type: ignore[import]

logger = logging.getLogger(__name__)

Gym = NewType("Gym", Any)
Env = NewType("Env", Any)
Sim = NewType("Sim", Any)
Robot = NewType("Robot", Any)
Viewer = NewType("Viewer", Any)
Args = NewType("Args", Any)


DRIVE_MODE = gymapi.DOF_MODE_EFFORT
# DRIVE_MODE = gymapi.DOF_MODE_POS

# Stiffness and damping are Kp and Kd parameters for the PD controller
# that drives the joints to the desired position.
STIFFNESS = 250.0
DAMPING = 10.0

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
    dof_ids: Dict[str, int] = None

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
    sim_params.physx.bounce_threshold_velocity = 0.5
    sim_params.physx.max_depenetration_velocity = 10.0
    sim_params.physx.max_gpu_contact_pairs = 2**24
    sim_params.physx.default_buffer_size_multiplier = 5
    sim_params.physx.contact_collection = gymapi.CC_ALL_SUBSTEPS

    sim_params.physx.num_threads = args.num_threads
    sim_params.physx.use_gpu = args.use_gpu

    sim_params.use_gpu_pipeline = False
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
    asset_options.collapse_fixed_joints = False
    asset_options.disable_gravity = False
    asset_options.fix_base_link = True
    asset_path = robot_urdf_path(legs_only=True)
    robot_asset = gym.load_urdf(sim, str(asset_path.parent), str(asset_path.name), asset_options)

    # Adds the robot to the environment.
    initial_pose = gymapi.Transform()
    initial_pose.p = gymapi.Vec3(0.0, 1.3, 0.0)
    # initial_pose.r = gymapi.Quat(0.5000, -0.4996, -0.5000, 0.5004)
    robot = gym.create_actor(env, robot_asset, initial_pose, "robot")

    # Configure DOF properties.
    props = gym.get_actor_dof_properties(env, robot)
    props["driveMode"] = DRIVE_MODE
    props["stiffness"].fill(STIFFNESS)
    props["damping"].fill(DAMPING)
    props["armature"].fill(ARMATURE)
    gym.set_actor_dof_properties(env, robot, props)

    # Look at the first environment.
    cam_pos = gymapi.Vec3(1, 2, 1.5)
    cam_target = gymapi.Vec3(0, 2, 1.5)
    gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

    # Gets tensors for the DOF states.
    dof_state_tensor = gym.acquire_dof_state_tensor(sim)
    gym.refresh_dof_state_tensor(sim)
    dof_state = gymtorch.wrap_tensor(dof_state_tensor)
    num_dof = len(Stompy.all_joints())
    dof_pos = dof_state.view(1, num_dof, 2)[..., 0]
    dof_vel = dof_state.view(1, num_dof, 2)[..., 1]

    #Resets the DOF positions to the starting positions.
    dof_vel[:] = 0.0
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
        dof_ids=dof_ids
    )


def pd_control(target_q, q, kp, dq, kd, default):
    """Calculates torques from position commands"""
    return kp * (target_q + default - q) - kd * dq


"""
Joint left hip pitch - Stiffness: 250.0, Damping: 10.0
Joint left hip yaw - Stiffness: 250.0, Damping: 10.0
Joint left hip roll - Stiffness: 150.0, Damping: 10.0
Joint left knee pitch - Stiffness: 250.0, Damping: 10.0
Joint left ankle pitch - Stiffness: 150.0, Damping: 10.0
Joint left shoulder pitch - Stiffness: 150.0, Damping: 10.0
Joint left shoulder yaw - Stiffness: 45.0, Damping: 10.0
Joint left shoulder roll - Stiffness: 45.0, Damping: 5.0
Joint left elbow pitch - Stiffness: 45.0, Damping: 5.0
Joint left wrist roll - Stiffness: 45.0, Damping: 5.0
Joint right hip pitch - Stiffness: 250.0, Damping: 10.0
Joint right hip yaw - Stiffness: 250.0, Damping: 10.0
Joint right hip roll - Stiffness: 150.0, Damping: 10.0
Joint right knee pitch - Stiffness: 250.0, Damping: 10.0
Joint right ankle pitch - Stiffness: 150.0, Damping: 10.0
Joint right shoulder pitch - Stiffness: 150.0, Damping: 10.0
Joint right shoulder yaw - Stiffness: 45.0, Damping: 10.0
Joint right shoulder roll - Stiffness: 45.0, Damping: 5.0
Joint right elbow pitch - Stiffness: 45.0, Damping: 5.0
Joint right wrist roll - Stiffness: 45.0, Damping: 5.0
"""

def run_gym(gym: GymParams, mode: Literal["one_at_a_time", "all_at_once"] = "all_at_once") -> None:
    torques = torch.zeros(1, len(Stompy.all_joints()))
    joint_id = 8
    test_duration = 10
    sin_freq = 2 * math.pi / test_duration
    tau_factor = 0.85
    tau_limit = np.array(list(Stompy.stiffness().values()) + list(Stompy.stiffness().values())) * tau_factor
    kps = tau_limit
    kds = np.array(list(Stompy.damping().values()) + list(Stompy.damping().values()))
    t0 = time.time()
    kd = 10
    kp = 250
    q = 0
    dq = 0
    default = Stompy.default_standing()["left knee pitch"]

    while not gym.gym.query_viewer_has_closed(gym.viewer):
        gym.gym.simulate(gym.sim)
        gym.gym.fetch_results(gym.sim, True)
        gym.gym.step_graphics(gym.sim)
        gym.gym.draw_viewer(gym.viewer, gym.sim, True)
        gym.gym.sync_frame_time(gym.sim)

        curr_time = time.time()
        sin_pos = torch.tensor(math.sin(sin_freq * (time.time() - t0)))
        # breakpoint()
        torques[:, joint_id] = pd_control(sin_pos, q, kp, dq, kd, default)
        gym.gym.set_dof_actuation_force_tensor(gym.sim, gymtorch.unwrap_tensor(torques))
        print(torques)
        # Prints the joint angles.
        joint_positions = gym.gym.get_actor_dof_states(gym.env, gym.robot, gymapi.STATE_ALL)


        dof_state_tensor = gym.gym.acquire_dof_state_tensor(gym.sim)
        gym.gym.refresh_dof_state_tensor(gym.sim)
        dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        num_dof = len(Stompy.all_joints())
        dof_pos = dof_state.view(1, num_dof, 2)[..., 0]
        dof_vel = dof_state.view(1, num_dof, 2)[..., 1]
        q = dof_pos[0, joint_id]
        dq = dof_vel[0, joint_id]
        print(joint_positions[8], q, dq)
        # for joint_name, (joint_position, joint_velocity) in zip(joints, joint_positions):
        #     print("Joint %s: %.3g %.3g", joint_name, joint_position, joint_velocity)

    gym.gym.destroy_viewer(gym.viewer)
    gym.gym.destroy_sim(gym.sim)


def main() -> None:
    gym = load_gym()
    run_gym(gym)


if __name__ == "__main__":
    main()
