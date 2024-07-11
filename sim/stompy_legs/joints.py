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
    hip_roll = "left hip roll"
    hip_yaw = "left hip yaw"
    hip_pitch = "left hip pitch"
    knee_pitch = "left knee pitch"
    ankle_pitch = "left ankle pitch"
    ankle_roll = "left ankle roll"


class RightLeg(Node):
    hip_roll = "right hip roll"
    hip_yaw = "right hip yaw"
    hip_pitch = "right hip pitch"
    knee_pitch = "right knee pitch"
    ankle_pitch = "right ankle pitch"
    ankle_roll = "right ankle roll"


class Legs(Node):
    left = LeftLeg()
    right = RightLeg()


class Stompy(Node):
    legs = Legs()

    @classmethod
    def default_positions(cls) -> Dict[str, float]:
        return {}

    @classmethod
    def default_standing(cls) -> Dict[str, float]:
        return {
            # legs
            Stompy.legs.left.hip_pitch: -0.12,
            Stompy.legs.left.hip_roll: 1.44,
            Stompy.legs.left.hip_yaw: -1.19,
            Stompy.legs.left.knee_pitch: -2.32,
            Stompy.legs.left.ankle_pitch: 0.56,
            Stompy.legs.left.ankle_roll: -2.64,
            Stompy.legs.right.hip_pitch: -4.33,
            Stompy.legs.right.hip_roll: 3.14,
            Stompy.legs.right.hip_yaw: -1.13,
            Stompy.legs.right.knee_pitch: -1.95,
            Stompy.legs.right.ankle_pitch: 0.62,
            Stompy.legs.right.ankle_roll: -2.64,
        }

    @classmethod
    def default_limits(cls) -> Dict[str, Dict[str, float]]:
        return {
            Stompy.legs.left.hip_pitch: {
                "lower": -1.32,
                "upper": 0.69,
            },
            Stompy.legs.left.hip_roll: {
                "lower": 1.13,
                "upper": 2.14,
            },
            Stompy.legs.left.hip_yaw: {
                "lower": -2.2,
                "upper": -1.19,
            },
            Stompy.legs.left.knee_pitch: {
                "lower": -3.14,
                "upper": -2.2,
            },
            Stompy.legs.left.ankle_pitch: {
                "lower": 0.44,
                "upper": 1.13,
            },
            Stompy.legs.left.ankle_roll: {
                "lower": -3.08,
                "upper": -2.26,
            },
            Stompy.legs.right.hip_pitch: {
                "lower": -5.00,
                "upper": -3.83,
            },
            Stompy.legs.right.hip_roll: {
                "lower": 2.39,
                "upper": 3.33,
            },
            Stompy.legs.right.hip_yaw: {
                "lower": -1.13,
                "upper": -0.69,
            },
            Stompy.legs.right.knee_pitch: {
                "lower": -1.95,
                "upper": -1.00,
            },
            Stompy.legs.right.ankle_pitch: {
                "lower": 0.00,
                "upper": 0.94,
            },
            Stompy.legs.right.ankle_roll: {
                "lower": -3.00,
                "upper": -2.14,
            },
        }


def print_joints() -> None:
    joints = Stompy.all_joints()
    assert len(joints) == len(set(joints)), "Duplicate joint names found!"
    print(Stompy())


if __name__ == "__main__":
    # python -m sim.stompy.joints
    print_joints()
