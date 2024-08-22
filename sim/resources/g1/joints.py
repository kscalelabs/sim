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
    hip_roll = "left_hip_roll_joint"
    hip_yaw = "left_hip_yaw_joint"
    hip_pitch = "left_hip_pitch_joint"
    knee_pitch = "left_knee_joint"
    ankle_pitch = "left_ankle_pitch_joint"
    ankle_roll = "left_ankle_roll_joint"


class RightLeg(Node):
    hip_roll = "right_hip_roll_joint"
    hip_yaw = "right_hip_yaw_joint"
    hip_pitch = "right_hip_pitch_joint"
    knee_pitch = "right_knee_joint"
    ankle_pitch = "right_ankle_pitch_joint"
    ankle_roll = "right_ankle_roll_joint"


class Legs(Node):
    left = LeftLeg()
    right = RightLeg()


class Robot(Node):
    legs = Legs()

    height = 1.1
    rotation = [0.0, 0.0, 0.7071068, 0.7071068]

    @classmethod
    def default_standing(cls) -> Dict[str, float]:
        return {  # = target angles [rad] when action = 0.0
            # left leg
            Robot.legs.left.hip_yaw: 0,
            Robot.legs.left.hip_roll: 0,
            Robot.legs.left.hip_pitch: -0.1,
            Robot.legs.left.knee_pitch: 0.3,
            Robot.legs.left.ankle_pitch: -0.2,
            Robot.legs.left.ankle_roll: 0.0,
            # right leg
            Robot.legs.right.hip_yaw: 0,
            Robot.legs.right.hip_roll: 0,
            Robot.legs.right.hip_pitch: -0.1,
            Robot.legs.right.knee_pitch: 0.3,
            Robot.legs.right.ankle_pitch: -0.2,
            Robot.legs.right.ankle_roll: 0.0,
        }

    # p_gains
    @classmethod
    def stiffness(cls) -> Dict[str, float]:
        return {
            "hip_pitch": 150,
            "hip_yaw": 150,
            "hip_roll": 150,
            "knee": 300,
            "ankle": 40,
        }

    # d_gains
    @classmethod
    def damping(cls) -> Dict[str, float]:
        return {
            "hip_pitch": 2,
            "hip_yaw": 2,
            "hip_roll": 2,
            "knee": 4,
            "ankle": 2,
        }


def print_joints() -> None:
    joints = Robot.all_joints()
    assert len(joints) == len(set(joints)), "Duplicate joint names found!"
    print(Robot())


if __name__ == "__main__":
    # python -m sim.Robot.joints
    print_joints()
