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


class LeftArm(Node):
    shoulder_yaw = "left_shoulder_yaw"
    shoulder_pitch = "left_shoulder_pitch"
    elbow_pitch = "left_elbow_yaw"  # FIXME: yaw vs pitch


class RightArm(Node):
    shoulder_yaw = "right_shoulder_yaw"
    shoulder_pitch = "right_shoulder_pitch"
    elbow_pitch = "right_elbow_yaw"  # FIXME: yaw vs pitch


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
    height = 0.21
    rotation = [0, 0, 0.707, 0.707]

    left_arm = LeftArm()
    right_arm = RightArm()
    legs = Legs()

    @classmethod
    def default_standing(cls) -> Dict[str, float]:
        return {
            # Legs
            cls.legs.left.hip_pitch: 0.0,
            cls.legs.left.hip_yaw: 0,
            cls.legs.left.hip_roll: 0,
            cls.legs.left.knee_pitch: 0,
            cls.legs.left.ankle_pitch: 0,
            cls.legs.right.hip_pitch: 0,
            cls.legs.right.hip_yaw: 0,
            cls.legs.right.hip_roll: 0,
            cls.legs.right.knee_pitch: 0,
            cls.legs.right.ankle_pitch: 0,
            # Arms (adjust as needed)
            cls.left_arm.shoulder_pitch: 0,
            cls.left_arm.shoulder_yaw: 0.3,
            cls.left_arm.elbow_pitch: -0.6,
            cls.right_arm.shoulder_pitch: 0,
            cls.right_arm.shoulder_yaw: -0.3,
            cls.right_arm.elbow_pitch: -0.6,
        }

    # FIXME: these limits are not correct
    @classmethod
    def default_limits(cls) -> Dict[str, Dict[str, float]]:
        return {
            # left arm
            Robot.left_arm.shoulder_pitch: {
                "lower": 2.04,
                "upper": 3.06,
            },
            Robot.left_arm.shoulder_yaw: {
                "lower": -1,
                "upper": 2,
            },
            Robot.left_arm.elbow_pitch: {
                "lower": -2.06,
                "upper": -1.08,
            },
            # right arm
            Robot.right_arm.shoulder_pitch: {
                "lower": 2.619,
                "upper": 3.621,
            },
            Robot.right_arm.shoulder_yaw: {
                "lower": -1.481,
                "upper": 1,
            },
            Robot.right_arm.elbow_pitch: {
                "lower": -3.819,
                "upper": 3.821,
            },
            Robot.legs.left.hip_pitch: {
                "lower": -1.64,
                "upper": 1.64,
            },
            Robot.legs.left.hip_roll: {
                "lower": -4.0,
                "upper": 1.0,
            },
            Robot.legs.left.hip_yaw: {
                "lower": 2.64,
                "upper": 5.64,
            },
            Robot.legs.left.knee_pitch: {
                "lower": -2.5,
                "upper": 0.5,
            },
            Robot.legs.left.ankle_pitch: {
                "lower": -0.3,
                "upper": 0.3,
            },
            # right leg
            Robot.legs.right.hip_pitch: {
                "lower": 0.05,
                "upper": 4.05,
            },
            Robot.legs.right.hip_roll: {
                "lower": 2.25,
                "upper": 4.49,
            },
            Robot.legs.right.hip_yaw: {
                "lower": 1.74,
                "upper": 4.74,
            },
            Robot.legs.right.knee_pitch: {
                "lower": -0.5,
                "upper": 2.5,
            },
            Robot.legs.right.ankle_pitch: {
                "lower": -0.3,
                "upper": 0.3,
            },
        }

    # p_gains
    @classmethod
    def stiffness(cls) -> Dict[str, float]:
        return {
            "hip_pitch": 5,
            "hip_yaw": 5,
            "hip_roll": 5,
            "knee_pitch": 5,
            "ankle_pitch": 5,
            "shoulder_pitch": 5,
            "shoulder_yaw": 5,
            "shoulder_roll": 5,
            "elbow_pitch": 5,
            "elbow_yaw": 5,
        }

    # d_gains
    @classmethod
    def damping(cls) -> Dict[str, float]:
        return {
            "hip_pitch": 0.3,
            "hip_yaw": 0.3,
            "hip_roll": 0.3,
            "knee_pitch": 0.3,
            "ankle_pitch": 0.3,
            "shoulder_pitch": 0.3,
            "shoulder_yaw": 0.3,
            "shoulder_roll": 0.3,
            "elbow_pitch": 0.3,
            "elbow_yaw": 0.3,
        }

    # pos_limits
    @classmethod
    def effort(cls) -> Dict[str, float]:
        return {
            "hip_pitch": 1,
            "hip_yaw": 1,
            "hip_roll": 1,
            "knee_pitch": 1,
            "ankle_pitch": 1,
            "shoulder_pitch": 1,
            "shoulder_yaw": 1,
            "shoulder_roll": 1,
            "elbow_pitch": 1,
            "elbow_yaw": 1,
        }

    # vel_limits
    @classmethod
    def velocity(cls) -> Dict[str, float]:
        return {
            "hip_pitch": 20,
            "hip_yaw": 20,
            "hip_roll": 20,
            "knee_pitch": 20,
            "ankle_pitch": 20,
            "shoulder_pitch": 20,
            "shoulder_yaw": 20,
            "shoulder_roll": 20,
            "elbow_pitch": 20,
            "elbow_yaw": 20,
        }

    @classmethod
    def friction(cls) -> Dict[str, float]:
        return {
            "hip_pitch": 0.05,
            "hip_yaw": 0.05,
            "hip_roll": 0.05,
            "knee_pitch": 0.05,
            "ankle_pitch": 0.05,
            "elbow_yaw": 0.05,
            "elbow_pitch": 0.05,
        }


def print_joints() -> None:
    joints = Robot.all_joints()
    assert len(joints) == len(set(joints)), "Duplicate joint names found!"
    print(Robot())


if __name__ == "__main__":
    # python -m sim.Robot.joints
    print_joints()
