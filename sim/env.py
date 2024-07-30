"""Defines common environment parameters."""

import os
from pathlib import Path

from dotenv import load_dotenv  # type: ignore

load_dotenv()


def model_dir() -> Path:
    return Path(os.environ.get("MODEL_DIR", "models"))


def run_dir() -> Path:
    return Path(os.environ.get("RUN_DIR", "runs"))


def stompy_urdf_path(legs_only: bool = False) -> Path:
    if legs_only:
        stompy_path = model_dir() / "robot_fixed.urdf"
    else:
        stompy_path = model_dir() / "robot_fixed.urdf"

    if not stompy_path.exists():
        raise FileNotFoundError(f"URDF file not found: {stompy_path}")

    return stompy_path.resolve()


def stompy_mjcf_path(legs_only: bool = False) -> Path:
    if legs_only:
        stompy_path = model_dir() / "robot_fixed.xml"
    else:
        stompy_path = model_dir() / "robot_fixed.xml"

    if not stompy_path.exists():
        raise FileNotFoundError(f"MJCF file not found: {stompy_path}")

    return stompy_path.resolve()
