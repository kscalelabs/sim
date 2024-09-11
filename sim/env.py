"""Defines common environment parameters."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def model_dir() -> Path:
    return Path(os.environ.get("MODEL_DIR", "sim/resources/"))


def run_dir() -> Path:
    return Path(os.environ.get("RUN_DIR", "runs"))


def robot_urdf_path(robot_name: str = "stompymini", legs_only: bool = False) -> Path:
    if legs_only:
        robot_path = model_dir() / robot_name / "robot_fixed.urdf"
    else:
        robot_path = model_dir() / robot_name / "robot_fixed.urdf"

    if not robot_path.exists():
        raise FileNotFoundError(f"URDF file not found: {robot_path}")

    return robot_path.resolve()


def robot_mjcf_path(robot_name: str = "stompymini", legs_only: bool = False) -> Path:
    if legs_only:
        robot_path = model_dir() / robot_name / "robot_fixed.xml"
    else:
        robot_path = model_dir() / robot_name / "robot_fixed.xml"

    if not robot_path.exists():
        raise FileNotFoundError(f"MJCF file not found: {robot_path}")

    return robot_path.resolve()
