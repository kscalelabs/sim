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
    hip_roll = "left_leg_roll_joint"
    hip_yaw = "left_leg_yaw_joint"
    hip_pitch = "left_leg_pitch_joint"
    knee_pitch = "left_knee_joint"
    ankle_pitch = "left_ankle_pitch_joint"
    ankle_roll = "left_ankle_roll_joint"


class RightLeg(Node):
    hip_roll = "right_leg_roll_joint"
    hip_yaw = "right_leg_yaw_joint"
    hip_pitch = "right_leg_pitch_joint"
    knee_pitch = "right_knee_joint"
    ankle_pitch = "right_ankle_pitch_joint"
    ankle_roll = "right_ankle_roll_joint"


class Legs(Node):
    left = LeftLeg()
    right = RightLeg()


class Robot(Node):
    height = 0.95
    rotation = [0, 0, 0, 1]

    legs = Legs()

    @classmethod
    def default_standing(cls) -> Dict[str, float]:
        return {
            # legs
            Robot.legs.left.hip_pitch: 0,
            Robot.legs.left.hip_roll: 0,
            Robot.legs.left.hip_yaw: 0,
            Robot.legs.left.knee_pitch: 0,
            Robot.legs.left.ankle_pitch: 0,
            Robot.legs.left.ankle_roll: 0,
            Robot.legs.right.hip_pitch: 0,
            Robot.legs.right.hip_roll: 0,
            Robot.legs.right.hip_yaw: 0,
            Robot.legs.right.knee_pitch: 0,
            Robot.legs.right.ankle_pitch: 0,
            Robot.legs.right.ankle_roll: 0,
        }

    @classmethod
    def default_limits(cls) -> Dict[str, Dict[str, float]]:
        return {
            # left leg
            Robot.legs.left.hip_pitch: {
                "lower": -1.57,
                "upper": 1.31,
            },
            Robot.legs.left.hip_roll: {
                "lower": -0.44,
                "upper": 1.57,
            },
            Robot.legs.left.hip_yaw: {
                "lower": -1.05,
                "upper": 1.05,
            },
            Robot.legs.left.knee_pitch: {
                "lower":  -1.05,
                "upper": 1.1,
            },
            Robot.legs.left.ankle_pitch: {
                "lower": -0.7,
                "upper": 0.87,
            },
            Robot.legs.left.ankle_roll: {
                "lower": -0.44,
                "upper": 0.44,
            },
            # right leg
            Robot.legs.right.hip_pitch: {
                "lower": -1.31,
                "upper": 1.57
            },
            Robot.legs.right.hip_roll: {
                "lower": -1.57,
                "upper": 0.44,
            },
            Robot.legs.right.hip_yaw: {
                "lower": -1.05,
                "upper": 1.05,
            },
            Robot.legs.right.knee_pitch: {
                "lower": -1.1,
                "upper": 1.05,
            },
            Robot.legs.right.ankle_pitch: {
                "lower": -0.87,
                "upper": 0.7,
            },
            Robot.legs.right.ankle_roll: {
                "lower": -0.44,
                "upper": 0.44,
            },
        }

    # p_gains
    @classmethod
    def stiffness(cls) -> Dict[str, float]:
        return {
            "leg_pitch": 350,
            "leg_yaw": 200,
            "leg_roll": 200,
            "knee": 350,
            "ankle_pitch": 15,
            "ankle_roll": 15,
        }

    # d_gains
    @classmethod
    def damping(cls) -> Dict[str, float]:
        return {
            "leg_pitch": 10,
            "leg_yaw": 10,
            "leg_roll": 10,
            "knee": 10,
            "ankle_pitch": 10,
            "ankle_roll": 10,
        }

    # pos_limits
    @classmethod
    def effort(cls) -> Dict[str, float]:
        return {
            "leg_pitch": 250,
            "leg_yaw": 100,
            "leg_roll": 100,
            "knee": 250,
            "ankle_pitch": 100,
            "ankle_roll": 100,
        }

    # vel_limits
    @classmethod
    def velocity(cls) -> Dict[str, float]:
        return {
            "leg_pitch": 12,
            "leg_yaw": 12,
            "leg_roll": 12,
            "knee": 12,
            "ankle_pitch": 12,
            "ankle_roll": 12,
        }

    @classmethod
    def friction(cls) -> Dict[str, float]:
        return {
            "leg_pitch": 0.0,
            "leg_yaw": 0.0,
            "leg_roll": 0.0,
            "knee": 0.0,
            "ankle_pitch": 0.0,
            "ankle_roll": 0.1,
        }


def print_joints() -> None:
    joints = Robot.all_joints()
    assert len(joints) == len(set(joints)), "Duplicate joint names found!"
    print(Robot())


if __name__ == "__main__":
    # python -m sim.Robot.joints
    print_joints()
