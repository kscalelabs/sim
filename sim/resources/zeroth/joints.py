"""Defines a more Pythonic interface for specifying the joint names.

The best way to re-generate this snippet for a new robot is to use the
`sim/scripts/print_joints.py` script. This script will print out a hierarchical
tree of the various joint names in the robot.
"""

import textwrap
from abc import ABC
from typing import Dict, List, Tuple, Union


class Node(ABC):
    @classmethod
    def children(cls) -> List["Union[Node, str]"]:
        return [
            attr
            for attr in (getattr(cls, attr) for attr in dir(cls) if not attr.startswith("__"))
            if isinstance(attr, (Node, str))
        ]

    @classmethod
    def joints(cls) -> List[str]:
        return [
            attr
            for attr in (getattr(cls, attr) for attr in dir(cls) if not attr.startswith("__"))
            if isinstance(attr, str)
        ]

    @classmethod
    def joints_motors(cls) -> List[Tuple[str, str]]:
        joint_names: List[Tuple[str, str]] = []
        for attr in dir(cls):
            if not attr.startswith("__"):
                attr2 = getattr(cls, attr)
                if isinstance(attr2, str):
                    joint_names.append((attr, attr2))

        return joint_names

    @classmethod
    def all_joints(cls) -> List[str]:
        leaves = cls.joints()
        for child in cls.children():
            if isinstance(child, Node):
                leaves.extend(child.all_joints())
        return leaves

    def __str__(self) -> str:
        parts = [str(child) for child in self.children()]
        parts_str = textwrap.indent("\n" + "\n".join(parts), "  ")
        return f"[{self.__class__.__name__}]{parts_str}"


class LeftLeg(Node):
    hip_roll = "left_hip_roll"
    hip_yaw = "left_hip_yaw"
    hip_pitch = "left_hip_pitch"
    knee_pitch = "left_knee_pitch"
    ankle_pitch = "left_ankle_pitch"


class RightLeg(Node):
    hip_roll = "right_hip_roll"
    hip_yaw = "right_hip_yaw"
    hip_pitch = "right_hip_pitch"
    knee_pitch = "right_knee_pitch"
    ankle_pitch = "right_ankle_pitch"


class Legs(Node):
    left = LeftLeg()
    right = RightLeg()


class Robot(Node):
    height = 0.42
    rotation = [0, 0, 0, 1.0]

    legs = Legs()

    @classmethod
    def default_standing(cls) -> Dict[str, float]:
        return {
            # Legs
            cls.legs.left.hip_pitch: -0.267,
            cls.legs.left.knee_pitch: -0.741,
            cls.legs.left.hip_yaw: 0,
            cls.legs.left.hip_roll: 0,
            cls.legs.left.ankle_pitch: 0.581,
            cls.legs.right.hip_pitch: -0.267,
            cls.legs.right.knee_pitch: 0.741,
            cls.legs.right.ankle_pitch: -0.581,
            cls.legs.right.hip_yaw: 0,
            cls.legs.right.hip_roll: 0,
        }

    @classmethod
    def default_limits(cls) -> Dict[str, Dict[str, float]]:
        return {
            Robot.legs.left.knee_pitch: {
                "lower": -1.57,
                "upper": 0,
            },
            Robot.legs.right.knee_pitch: {
                "lower": 0,
                "upper": 1.57,
            }
        }

    # p_gains
    @classmethod
    def stiffness(cls) -> Dict[str, float]:
        return {
            "hip_pitch": 17.681462808698132,
            "hip_yaw": 17.681462808698132,
            "hip_roll": 17.681462808698132,
            "knee_pitch": 17.681462808698132,
            "ankle_pitch": 17.681462808698132,
        }

    # d_gains
    @classmethod
    def damping(cls) -> Dict[str, float]:
        return {
            "hip_pitch": 0.5354656169048285,
            "hip_yaw": 0.5354656169048285,
            "hip_roll": 0.5354656169048285,
            "knee_pitch": 0.5354656169048285,
            "ankle_pitch": 0.5354656169048285,
        }

    # pos_limits
    @classmethod
    def effort(cls) -> Dict[str, float]:
        return {
            "hip_pitch": 10,
            "hip_yaw": 10,
            "hip_roll": 10,
            "knee_pitch": 10,
            "ankle_pitch": 10,
        }

    # vel_limits
    @classmethod
    def velocity(cls) -> Dict[str, float]:
        return {
            "hip_pitch": 10,
            "hip_yaw": 10,
            "hip_roll": 10,
            "knee_pitch": 10,
            "ankle_pitch": 10,
        }

    @classmethod
    def friction(cls) -> Dict[str, float]:
        return {
            "ankle_pitch": 0.01,
        }


def print_joints() -> None:
    joints = Robot.all_joints()
    assert len(joints) == len(set(joints)), "Duplicate joint names found!"
    print(Robot())


if __name__ == "__main__":
    # python -m sim.Robot.joints
    print_joints()