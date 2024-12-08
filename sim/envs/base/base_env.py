from abc import ABC, abstractmethod
from typing import Any, Tuple


class Env(ABC):
    """Minimal base environment class for RL."""
    
    @abstractmethod
    def step(self, action: Any) -> Tuple[Any, float, bool, dict]:
        """Run one timestep of the environment.
        Returns: observation, reward, done, info"""
        pass
    
    @abstractmethod
    def reset(self) -> Any:
        """Reset the environment to initial state."""
        pass
    
    def render(self) -> None:
        """Optional method to render the environment."""
        pass
    
    def close(self) -> None:
        """Clean up resources."""
        pass
