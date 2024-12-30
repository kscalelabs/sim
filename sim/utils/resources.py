import importlib
from typing import Any


def load_embodiment(embodiment: str) -> Any:  # noqa: ANN401
    # Dynamically import embodiment
    module_name = f"sim.resources.{embodiment}.joints"
    module = importlib.import_module(module_name)
    robot = getattr(module, "Robot")
    return robot
