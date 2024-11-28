"""Defines common environment parameters."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def model_dir(robotname: str) -> Path:
    print(os.getcwd())

    model_dir = Path(os.environ.get("MODEL_DIR", "sim/resources/" + robotname))
    return model_dir


def run_dir() -> Path:
    return Path(os.environ.get("RUN_DIR", "runs"))


def robot_urdf_path(robotname: str, legs_only: bool = True) -> Path:
    print(f"robot_urdf_path({robotname=}, {legs_only=})")
    filename = "robot_fixed.urdf" if legs_only else "robot.urdf"
    robot_path = model_dir(robotname) / filename
    if not robot_path.exists():
        raise FileNotFoundError(f"URDF file not found: {robot_path}")
    print(f"robot_path={robot_path}")
    return robot_path.resolve()


def robot_mjcf_path(robotname: str, legs_only: bool = True) -> Path:
    print(f"robot_mjcf_path({robotname=}, {legs_only=})")
    filename = "robot_fixed.xml" if legs_only else "robot.xml"
    robot_path = model_dir(robotname) / filename
    if not robot_path.exists():
        raise FileNotFoundError(f"MJCF file not found: {robot_path}")
    print(f"robot_path={robot_path}")
    return robot_path.resolve()
