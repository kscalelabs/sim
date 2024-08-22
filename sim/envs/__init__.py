"""Registers the tasks in the task registry.

For other people who might be looking at this in the future - my preferred way
of doing config management is to use dataclasses (see the `mlfab` or `xax`
packages for examples of what I mean). This plays a lot better with type
checkers and VSCode. I am just doing it this way to get something working
quickly.
"""

# fmt: off
# import isaacgym # isort:skip
# import torch # isort:skip
from .base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
from .humanoids.dora_config import DoraCfg, DoraCfgPPO
from .humanoids.dora_env import DoraFreeEnv
from .humanoids.g1_config import G1Cfg, G1CfgPPO
from .humanoids.g1_env import G1FreeEnv
from .humanoids.h1_config import H1Cfg, H1CfgPPO
from .humanoids.h1_env import H1FreeEnv
from .humanoids.stompymini_config import MiniCfg, MiniCfgPPO
from .humanoids.stompymini_env import MiniFreeEnv

# fmt: on


def register_tasks() -> None:
    """Registers the tasks in the task registry.

    This should take place in a separate function to make the import order
    explicit (meaning, avoiding importing torch after IsaacGym).
    """
    from utils.task_registry import task_registry

    task_registry.register("dora_ppo", DoraFreeEnv, DoraCfg(), DoraCfgPPO())
    task_registry.register("h1", H1FreeEnv, H1Cfg(), H1CfgPPO())
    task_registry.register("g1", G1FreeEnv, G1Cfg(), G1CfgPPO())
    task_registry.register("stompymini", MiniFreeEnv, MiniCfg(), MiniCfgPPO())


breakpoint()
register_tasks()
