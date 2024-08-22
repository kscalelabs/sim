import os

__version__ = "0.0.1"

LEGGED_GYM_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
LEGGED_GYM_ENVS_DIR = os.path.join(LEGGED_GYM_ROOT_DIR, "envs")
