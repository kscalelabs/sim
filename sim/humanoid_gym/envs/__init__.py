"""Registers the tasks in the task registry.

For other people who might be looking at this in the future - my preferred way
of doing config management is to use dataclasses (see the `mlfab` or `xax`
packages for examples of what I mean). This plays a lot better with type
checkers and VSCode. I am just doing it this way to get something working
quickly.
"""

# fmt: off
import isaacgym # isort:skip
import torch # isort:skip
from .g1_config import G1Cfg, G1CfgPPO
from .g1_env import G1FreeEnv
from .h1_config import H1Cfg, H1CfgPPO
from .h1_env import H1FreeEnv

# fmt: on
from .only_legs_config import OnlyLegsCfg, OnlyLegsCfgPPO
from .only_legs_env import OnlyLegsFreeEnv
from .stompymini_config import MiniCfg, MiniCfgPPO
from .stompymini_env import MiniFreeEnv


def register_tasks() -> None:
    """Registers the tasks in the task registry.

    This should take place in a separate function to make the import order
    explicit (meaning, avoiding importing torch after IsaacGym).
    """
    from humanoid.utils.task_registry import task_registry

    task_registry.register("h1", H1FreeEnv, H1Cfg(), H1CfgPPO())
    task_registry.register("g1", G1FreeEnv, G1Cfg(), G1CfgPPO())
    task_registry.register("only_legs_ppo", OnlyLegsFreeEnv, OnlyLegsCfg(), OnlyLegsCfgPPO())
    task_registry.register("stompymini", MiniFreeEnv, MiniCfg(), MiniCfgPPO())


register_tasks()
