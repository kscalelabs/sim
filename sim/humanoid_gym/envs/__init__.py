"""Registers the tasks in the task registry.

For other people who might be looking at this in the future - my preferred way
of doing config management is to use dataclasses (see the `mlfab` or `xax`
packages for examples of what I mean). This plays a lot better with type
checkers and VSCode. I am just doing it this way to get something working
quickly.
"""

# fmt: off
import isaacgym # isort:skip
import torch #isort:skip
# fmt: on
from .getup_config import GetupCfg, GetupCfgPPO
from .getup_env import GetupFreeEnv
from .only_legs_config import OnlyLegsCfg, OnlyLegsCfgPPO
from .only_legs_env import OnlyLegsFreeEnv
from .stompy_config import StompyCfg, StompyCfgPPO
from .stompy_env import StompyFreeEnv


def register_tasks() -> None:
    """Registers the tasks in the task registry.

    This should take place in a separate function to make the import order
    explicit (meaning, avoiding importing torch after IsaacGym).
    """
    from humanoid.utils.task_registry import task_registry

    task_registry.register("stompy_ppo", StompyFreeEnv, StompyCfg(), StompyCfgPPO())
    task_registry.register("getup_ppo", GetupFreeEnv, GetupCfg(), GetupCfgPPO())
    task_registry.register("only_legs_ppo", OnlyLegsFreeEnv, OnlyLegsCfg(), OnlyLegsCfgPPO())


register_tasks()
