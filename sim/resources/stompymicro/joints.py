"""Defines a more Pythonic interface for specifying the joint names.

The best way to re-generate this snippet for a new robot is to use the
`sim/scripts/print_joints.py` script. This script will print out a hierarchical
tree of the various joint names in the robot.
"""

import math
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


class RightArm(Node):
    shoulder_pitch = "right_shoulder_pitch"
    shoulder_yaw = "right_shoulder_yaw"
    elbow_yaw = "right_elbow_yaw"


class LeftArm(Node):
    shoulder_pitch = "left_shoulder_pitch"
    shoulder_yaw = "left_shoulder_yaw"
    elbow_yaw = "left_elbow_yaw"


class RightLeg(Node):
    hip_pitch = "right_hip_pitch"
    hip_yaw = "right_hip_yaw"
    hip_roll = "right_hip_roll"
    knee_pitch = "right_knee_pitch"
    ankle_pitch = "right_ankle_pitch"


class LeftLeg(Node):
    hip_pitch = "left_hip_pitch"
    hip_yaw = "left_hip_yaw"
    hip_roll = "left_hip_roll"
    ankle_pitch = "left_ankle_pitch"
    knee_pitch = "left_knee_pitch"


class Legs(Node):
    right = RightLeg()
    left = LeftLeg()


class Arms(Node):
    right = RightArm()
    left = LeftArm()


def get_facing_direction_quaternion(angle_degrees: float) -> List[float]:
    theta = angle_degrees * (math.pi / 180)
    half_theta = theta / 2
    return [0, 0, math.sin(half_theta), math.cos(half_theta)]


class Robot(Node):
    height = 0.32
    angle = 0
    rotation = get_facing_direction_quaternion(angle)
    print(f"Rotation: {rotation}")

    legs_only = False

    legs = Legs()
    arms = Arms() if not legs_only else None

    @classmethod
    def default_standing(cls) -> Dict[str, float]:
        legs = {
            cls.legs.left.hip_pitch: 0.23,
            cls.legs.left.knee_pitch: -0.741,
            cls.legs.left.hip_yaw: 0,
            cls.legs.left.hip_roll: 0,
            cls.legs.left.ankle_pitch: -0.5,
            cls.legs.right.hip_pitch: -0.23,
            cls.legs.right.knee_pitch: 0.741,
            cls.legs.right.ankle_pitch: 0.5,
            cls.legs.right.hip_yaw: 0,
            cls.legs.right.hip_roll: 0,
        }

        if cls.arms is None:
            return legs

        arms = {
            cls.arms.left.shoulder_pitch: 0.0,
            cls.arms.left.shoulder_yaw: 0.0,
            cls.arms.left.elbow_yaw: 0.0,
            cls.arms.right.shoulder_pitch: 0.0,
            cls.arms.right.shoulder_yaw: 0.0,
            cls.arms.right.elbow_yaw: 0.0,
        }
        return {**legs, **arms}

    @classmethod
    def default_limits(cls) -> Dict[str, Dict[str, float]]:
        legs = {
            # Left Leg
            cls.legs.left.hip_pitch: {
                "lower": -1.5707963,
                "upper": 1.5707963,
            },
            cls.legs.left.hip_yaw: {
                "lower": -1.5707963,
                "upper": 0.087266463,
            },
            cls.legs.left.hip_roll: {
                "lower": -0.78539816,
                "upper": 0.78539816,
            },
            cls.legs.left.knee_pitch: {
                "lower": -1.0471976,
                "upper": 0,
            },
            cls.legs.left.ankle_pitch: {
                "lower": -1.5707963,
                "upper": 1.5707963,
            },
            # Right Leg
            cls.legs.right.hip_pitch: {
                "lower": -1.5707963,
                "upper": 1.5707963,
            },
            cls.legs.right.hip_yaw: {
                "lower": -0.087266463,
                "upper": 1.5707963,
            },
            cls.legs.right.hip_roll: {
                "lower": -0.78539816,
                "upper": 0.78539816,
            },
            cls.legs.right.knee_pitch: {
                "lower": 0,
                "upper": 1.0471976,
            },
            cls.legs.right.ankle_pitch: {
                "lower": -1.5707963,
                "upper": 1.5707963,
            },
        }

        if cls.arms is None:
            return legs

        arms = {
            # Left Arm
            cls.arms.left.shoulder_pitch: {
                "lower": -1.7453293,
                "upper": 1.7453293,
            },
            cls.arms.left.shoulder_yaw: {
                "lower": -0.43633231,
                "upper": 1.5707963,
            },
            cls.arms.left.elbow_yaw: {
                "lower": -1.5707963,
                "upper": 1.5707963,
            },
            # Right Arm
            cls.arms.right.shoulder_pitch: {
                "lower": -1.7453293,
                "upper": 1.7453293,
            },
            cls.arms.right.shoulder_yaw: {
                "lower": -1.134464,
                "upper": 0.87266463,
            },
            cls.arms.right.elbow_yaw: {
                "lower": -1.5707963,
                "upper": 1.5707963,
            },
        }
        return {**legs, **arms}

    # p_gains
    @classmethod
    def stiffness(cls) -> Dict[str, float]:
        legs = {
            "hip_pitch": 17.681462808698132,
            "hip_yaw": 17.681462808698132,
            "hip_roll": 17.681462808698132,
            "knee_pitch": 17.681462808698132,
            "ankle_pitch": 17.681462808698132,
        }

        if cls.arms is None:
            return legs

        arms = {
            "shoulder_pitch": 5.0,
            "shoulder_yaw": 3.75,
            "elbow_yaw": 3.5,
        }
        return {**legs, **arms}

    # d_gains
    @classmethod
    def damping(cls) -> Dict[str, float]:
        legs = {
            "hip_pitch": 0.5354656169048285,
            "hip_yaw": 0.5354656169048285,
            "hip_roll": 0.5354656169048285,
            "knee_pitch": 0.5354656169048285,
            "ankle_pitch": 0.5354656169048285,
        }

        if cls.arms is None:
            return legs

        arms = {
            "shoulder_pitch": 0.3,
            "shoulder_yaw": 0.3,
            "elbow_yaw": 0.2,
        }
        return {**legs, **arms}

    # pos_limits
    @classmethod
    def effort(cls) -> Dict[str, float]:
        legs = {
            "hip_pitch": 10,
            "hip_yaw": 10,
            "hip_roll": 10,
            "knee_pitch": 10,
            "ankle_pitch": 10,
        }

        if cls.arms is None:
            return legs

        arms = {
            "shoulder_pitch": 80,
            "shoulder_yaw": 80,
            "elbow_yaw": 80,
        }
        return {**legs, **arms}

    # vel_limits
    @classmethod
    def velocity(cls) -> Dict[str, float]:
        legs = {
            "hip_pitch": 5,
            "hip_yaw": 5,
            "hip_roll": 5,
            "knee_pitch": 5,
            "ankle_pitch": 5,
        }

        if cls.arms is None:
            return legs

        arms = {
            "shoulder_pitch": 5,
            "shoulder_yaw": 5,
            "elbow_yaw": 5,
        }
        return {**legs, **arms}

    @classmethod
    def friction(cls) -> Dict[str, float]:
        legs = {
            "hip_pitch": 0.05,
            "hip_yaw": 0.05,
            "hip_roll": 0.05,
            "knee_pitch": 0.05,
            "ankle_pitch": 0.05,
        }

        if cls.arms is None:
            return legs

        arms = {
            "shoulder_pitch": 0.05,
            "shoulder_yaw": 0.05,
            "elbow_yaw": 0.05,
        }
        return {**legs, **arms}


def print_joints() -> None:
    joints = Robot.all_joints()
    assert len(joints) == len(set(joints)), "Duplicate joint names found!"
    print(Robot())
    print(f"\nArms are {'disabled' if Robot.legs_only else 'enabled'}")
    print(f"Number of joints: {len(joints)}")


if __name__ == "__main__":
    # python -m sim.resources.[robotname].joints
    print_joints()
