"""Registers the tasks in the task registry.

For other people who might be looking at this in the future - my preferred way
of doing config management is to use dataclasses (see the `mlfab` or `xax`
packages for examples of what I mean). This plays a lot better with type
checkers and VSCode. I am just doing it this way to get something working
quickly.
"""

from .humanoid_config import StompyCfg, StompyPPO
from .humanoid_env import StompyFreeEnv


def register_tasks() -> None:
    """Registers the tasks in the task registry.

    This should take place in a separate function to make the import order
    explicit (meaning, avoiding importing torch after IsaacGym).
    """
    from humanoid.utils.task_registry import task_registry

    task_registry.register("humanoid_ppo", StompyFreeEnv, StompyCfg(), StompyPPO())


register_tasks()
