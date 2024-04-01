"""Script to drop a URDF in Isaac Gym."""

import argparse

from pathlib import Path
from sim.isaacgym import gymutil, gymtorch, gymapi
import torch
import mlfab


def get_sim_params(num_threads: int, use_gpu: bool) -> gymapi.SimParams:
    sim_params = gymapi.SimParams()
    sim_params.substeps = 2
    sim_params.dt = 1.0 / 60.0

    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 1

    sim_params.physx.num_threads = num_threads
    sim_params.physx.use_gpu = use_gpu

    return sim_params



def main() -> None:
    parser = argparse.ArgumentParser(description="Drop a URDF in Isaac Gym.")
    parser.add_argument("urdf_path", help="Path to the URDF file to load")
    args = parser.parse_args()

    mlfab.configure_logging()

    # Verifies that the URDF exists.
    urdf_path = Path(args.urdf_path)
    if not urdf_path.exists():
        raise FileNotFoundError(f"URDF file not found: {urdf_path}")

    sim_params = get_sim_params(args.num_threads, args.use_gpu)


if __name__ == "__main__":
    # python -m sim.scripts.drop_urdf
    main()
