"""Defines common environment parameters."""

import os
from pathlib import Path


def model_dir() -> Path:
    return Path(os.environ.get("MODEL_DIR", "models"))
