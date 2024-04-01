"""Defines training code for Stompy."""

import logging
import warnings
from pathlib import Path
from typing import Any

from isaacgym import gymapi, gymutil

from sim.env import model_dir

logger = logging.getLogger(__name__)

Gym = Any


def get_stompy_path() -> Path:
    urdf_path = model_dir() / "robots" / "stompy" / "robot.urdf"
    if not urdf_path.exists():
        raise FileNotFoundError(f"URDF file not found: {urdf_path}")
    return urdf_path


def load_gym() -> Gym:
    # Initialize gym.
    gym = gymapi.acquire_gym()

    # Parse arguments.
    args = gymutil.parse_arguments(description="Joint control methods")

    # Sets the simulation parameters.
    sim_params = gymapi.SimParams()
    sim_params.substeps = 2
    sim_params.dt = 1.0 / 60.0

    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 1

    sim_params.physx.num_threads = args.num_threads
    sim_params.physx.use_gpu = args.use_gpu

    sim_params.use_gpu_pipeline = False
    if args.use_gpu_pipeline:
        warnings.warn("Forcing CPU pipeline.")

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

    return gym


def main() -> None:
    args = gymutil.parse_arguments(description="Joint control methods")

    stompy_path = get_stompy_path()


if __name__ == "__main__":
    # python -m sim.stompy
    main()
