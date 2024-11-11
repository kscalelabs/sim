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
    elbow_yaw = "left_elbow_yaw"


class RightArm(Node):
    shoulder_yaw = "right_shoulder_yaw"
    shoulder_pitch = "right_shoulder_pitch"
    elbow_yaw = "right_elbow_yaw"


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


class Arms(Node):
    left = LeftArm()
    right = RightArm()


class Robot(Node):
    USE_ARMS = False  # Set to True to enable active arms

    height = 0.32
    rotation = [0, 0, 0.707, 0.707]

    arms = Arms()
    legs = Legs()

    @classmethod
    def default_standing(cls) -> Dict[str, float]:
        base_dict = {
            # Legs
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

        if cls.USE_ARMS:
            base_dict.update(
                {
                    cls.arms.left.shoulder_pitch: 0.0,
                    cls.arms.left.shoulder_yaw: 0.0,
                    cls.arms.left.elbow_yaw: 0.0,
                    cls.arms.right.shoulder_pitch: 0.0,
                    cls.arms.right.shoulder_yaw: 0.0,
                    cls.arms.right.elbow_yaw: 0.0,
                }
            )

        return base_dict

    @classmethod
    def default_limits(cls) -> Dict[str, Dict[str, float]]:
        base_limits = {
            # Left Leg
            Robot.legs.left.hip_pitch: {
                "lower": -1.5707963,
                "upper": 1.5707963,
            },
            Robot.legs.left.hip_yaw: {
                "lower": -1.5707963,
                "upper": 0.087266463,
            },
            Robot.legs.left.hip_roll: {
                "lower": -0.78539816,
                "upper": 0.78539816,
            },
            Robot.legs.left.knee_pitch: {
                "lower": -1.0471976,
                "upper": 0,
            },
            Robot.legs.left.ankle_pitch: {
                "lower": -1.5707963,
                "upper": 1.5707963,
            },
            # Right Leg
            Robot.legs.right.hip_pitch: {
                "lower": -1.5707963,
                "upper": 1.5707963,
            },
            Robot.legs.right.hip_yaw: {
                "lower": -0.087266463,
                "upper": 1.5707963,
            },
            Robot.legs.right.hip_roll: {
                "lower": -0.78539816,
                "upper": 0.78539816,
            },
            Robot.legs.right.knee_pitch: {
                "lower": 0,
                "upper": 1.0471976,
            },
            Robot.legs.right.ankle_pitch: {
                "lower": -1.5707963,
                "upper": 1.5707963,
            },
        }

        if cls.USE_ARMS:
            arm_limits = {
                # Left Arm
                Robot.arms.left.shoulder_pitch: {
                    "lower": -1.7453293,
                    "upper": 1.7453293,
                },
                Robot.arms.left.shoulder_yaw: {
                    "lower": -0.43633231,
                    "upper": 1.5707963,
                },
                Robot.arms.left.elbow_yaw: {
                    "lower": -1.5707963,
                    "upper": 1.5707963,
                },
                # Right Arm
                Robot.arms.right.shoulder_pitch: {
                    "lower": -1.7453293,
                    "upper": 1.7453293,
                },
                Robot.arms.right.shoulder_yaw: {
                    "lower": -1.134464,
                    "upper": 0.87266463,
                },
                Robot.arms.right.elbow_yaw: {
                    "lower": -1.5707963,
                    "upper": 1.5707963,
                },
            }
            base_limits.update(arm_limits)

        return base_limits

    # p_gains
    @classmethod
    def stiffness(cls) -> Dict[str, float]:
        base_dict = {
            "hip_pitch": 17.681462808698132,
            "hip_yaw": 17.681462808698132,
            "hip_roll": 17.681462808698132,
            "knee_pitch": 17.681462808698132,
            "ankle_pitch": 17.681462808698132,
        }

        if cls.USE_ARMS:
            base_dict.update(
                {
                    "shoulder_pitch": 5.0,
                    "shoulder_yaw": 5.0,
                    "elbow_yaw": 5.0,
                }
            )

        return base_dict

    # d_gains
    @classmethod
    def damping(cls) -> Dict[str, float]:
        base_dict = {
            "hip_pitch": 0.5354656169048285,
            "hip_yaw": 0.5354656169048285,
            "hip_roll": 0.5354656169048285,
            "knee_pitch": 0.5354656169048285,
            "ankle_pitch": 0.5354656169048285,
        }

        if cls.USE_ARMS:
            base_dict.update(
                {
                    "shoulder_pitch": 0.3,
                    "shoulder_yaw": 0.3,
                    "elbow_yaw": 0.3,
                }
            )

        return base_dict

    # pos_limits
    @classmethod
    def effort(cls) -> Dict[str, float]:
        base_dict = {
            "hip_pitch": 10,
            "hip_yaw": 10,
            "hip_roll": 10,
            "knee_pitch": 10,
            "ankle_pitch": 10,
        }

        if cls.USE_ARMS:
            base_dict.update(
                {
                    "shoulder_pitch": 80,
                    "shoulder_yaw": 80,
                    "elbow_yaw": 80,
                }
            )

        return base_dict

    # vel_limits
    @classmethod
    def velocity(cls) -> Dict[str, float]:
        base_dict = {
            "hip_pitch": 10,
            "hip_yaw": 10,
            "hip_roll": 10,
            "knee_pitch": 10,
            "ankle_pitch": 10,
        }

        if cls.USE_ARMS:
            base_dict.update(
                {
                    "shoulder_pitch": 5,
                    "shoulder_yaw": 5,
                    "elbow_yaw": 5,
                }
            )

        return base_dict

    @classmethod
    def friction(cls) -> Dict[str, float]:
        base_dict = {
            "hip_pitch": 0.05,
            "hip_yaw": 0.05,
            "hip_roll": 0.05,
            "knee_pitch": 0.05,
            "ankle_pitch": 0.05,
        }

        if cls.USE_ARMS:
            base_dict.update(
                {
                    "shoulder_pitch": 0.05,
                    "shoulder_yaw": 0.05,
                    "elbow_yaw": 0.05,
                }
            )

        return base_dict


def print_joints() -> None:
    joints = Robot.all_joints()
    assert len(joints) == len(set(joints)), "Duplicate joint names found!"
    print(Robot())
    print(f"\nArms are {'enabled' if Robot.USE_ARMS else 'disabled'}")
    print(f"Number of joints: {len(joints)}")


if __name__ == "__main__":
    # python -m sim.Robot.joints
    print_joints()
