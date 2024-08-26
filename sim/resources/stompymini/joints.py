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


class LeftHand(Node):
    wrist_roll = "left wrist roll"


class LeftArm(Node):
    shoulder_yaw = "left shoulder yaw"
    shoulder_pitch = "left shoulder pitch"
    shoulder_roll = "left shoulder roll"
    elbow_pitch = "left elbow pitch"
    hand = LeftHand()


class RightHand(Node):
    wrist_roll = "right wrist roll"


class RightArm(Node):
    shoulder_yaw = "right shoulder yaw"
    shoulder_pitch = "right shoulder pitch"
    shoulder_roll = "right shoulder roll"
    elbow_pitch = "right elbow pitch"
    hand = RightHand()


class LeftLeg(Node):
    hip_roll = "left hip roll"
    hip_yaw = "left hip yaw"
    hip_pitch = "left hip pitch"
    knee_pitch = "left knee pitch"
    ankle_pitch = "left ankle pitch"


class RightLeg(Node):
    hip_roll = "right hip roll"
    hip_yaw = "right hip yaw"
    hip_pitch = "right hip pitch"
    knee_pitch = "right knee pitch"
    ankle_pitch = "right ankle pitch"


class Legs(Node):
    left = LeftLeg()
    right = RightLeg()


class Robot(Node):
    height = 0.82
    rotation = [0.5000, -0.4996, -0.5000, 0.5004]
    collision_links = [
        "lower_half_assembly_1_left_leg_1_foot_pad_1_simple",
        "lower_half_assembly_1_right_leg_1_foot_pad_1_simple",
    ]

    left_arm = LeftArm()
    right_arm = RightArm()
    legs = Legs()

    @classmethod
    def default_standing(cls) -> Dict[str, float]:
        return {
            # arms
            Robot.left_arm.shoulder_pitch: -1.02,
            Robot.left_arm.shoulder_yaw: 1.38,
            Robot.left_arm.shoulder_roll: -3.24,
            Robot.left_arm.elbow_pitch: 1.2,
            Robot.left_arm.hand.wrist_roll: 0,
            Robot.right_arm.shoulder_pitch: 3.12,
            Robot.right_arm.shoulder_yaw: -1.98,
            Robot.right_arm.shoulder_roll: -1.38,
            Robot.right_arm.elbow_pitch: 1.32,
            # hands
            Robot.right_arm.hand.wrist_roll: 0,
            # legs
            Robot.legs.left.hip_pitch: -0.28,
            Robot.legs.left.hip_roll: 1.5,
            Robot.legs.left.hip_yaw: 1.62,
            Robot.legs.left.knee_pitch: 1,
            Robot.legs.left.ankle_pitch: -2.2,
            Robot.legs.right.hip_pitch: 3.55,
            Robot.legs.right.hip_roll: 3.18,
            Robot.legs.right.hip_yaw: 3.24,
            Robot.legs.right.knee_pitch: -1,
            Robot.legs.right.ankle_pitch: 0.42,
        }

    @classmethod
    def default_limits(cls) -> Dict[str, Dict[str, float]]:
        return {
            # left arm
            Robot.left_arm.shoulder_pitch: {
                "lower": -1.021,
                "upper": -1.019,
            },
            Robot.left_arm.shoulder_yaw: {
                "lower": 1.379,
                "upper": 1.381,
            },
            Robot.left_arm.shoulder_roll: {
                "lower": -3.241,
                "upper": -3.239,
            },
            Robot.left_arm.elbow_pitch: {
                "lower": 1.199,
                "upper": 1.201,
            },
            Robot.left_arm.hand.wrist_roll: {
                "lower": -0.001,
                "upper": 0.001,
            },
            # right arm
            Robot.right_arm.shoulder_pitch: {
                "lower": 3.119,
                "upper": 3.121,
            },
            Robot.right_arm.shoulder_yaw: {
                "lower": -1.981,
                "upper": -1.979,
            },
            Robot.right_arm.shoulder_roll: {
                "lower": -1.381,
                "upper": -1.379,
            },
            Robot.right_arm.elbow_pitch: {
                "lower": 1.319,
                "upper": 1.321,
            },
            Robot.right_arm.hand.wrist_roll: {
                "lower": -0.001,
                "upper": 0.001,
            },
            # left leg
            Robot.legs.left.hip_pitch: {
                "lower": -1.28,
                "upper": 0.72,
            },
            Robot.legs.left.hip_roll: {
                "lower": 0.5,
                "upper": 2.5,
            },
            Robot.legs.left.hip_yaw: {
                "lower": 0.62,
                "upper": 2.62,
            },
            Robot.legs.left.knee_pitch: {
                "lower": 0,
                "upper": 2,
            },
            Robot.legs.left.ankle_pitch: {
                "lower": -2.8,
                "upper": -0.8,
            },
            # right leg
            Robot.legs.right.hip_pitch: {
                "lower": 2.55,
                "upper": 4.55,
            },
            Robot.legs.right.hip_roll: {
                "lower": 2.18,
                "upper": 4.18,
            },
            Robot.legs.right.hip_yaw: {
                "lower": 2.24,
                "upper": 4.24,
            },
            Robot.legs.right.knee_pitch: {
                "lower": -2,
                "upper": 0,
            },
            Robot.legs.right.ankle_pitch: {
                "lower": -0.58,
                "upper": 1.42,
            },
        }

    # p_gains
    @classmethod
    def stiffness(cls) -> Dict[str, float]:
        return {
            "hip pitch": 250,
            "hip yaw": 250,
            "hip roll": 150,
            "knee pitch": 250,
            "ankle pitch": 150,
            "shoulder pitch": 150,
            "shoulder yaw": 45,
            "shoulder roll": 45,
            "elbow pitch": 45,
            "wrist roll": 45,
        }

    # d_gains
    @classmethod
    def damping(cls) -> Dict[str, float]:
        return {
            "hip pitch": 10,
            "hip yaw": 10,
            "hip roll": 10,
            "knee pitch": 10,
            "ankle pitch": 10,
            "shoulder pitch": 10,
            "shoulder yaw": 10,
            "shoulder roll": 5,
            "elbow pitch": 5,
            "wrist roll": 5,
        }

    # pos_limits
    @classmethod
    def effort(cls) -> Dict[str, float]:
        return {
            "hip pitch": 120,
            "hip yaw": 120,
            "hip roll": 17,
            "knee pitch": 120,
            "ankle pitch": 17,
            "shoulder pitch": 120,
            "shoulder yaw": 17,
            "shoulder roll": 17,
            "elbow pitch": 17,
            "wrist roll": 17,
        }

    # vel_limits
    @classmethod
    def velocity(cls) -> Dict[str, float]:
        return {
            "hip pitch": 40,
            "hip yaw": 40,
            "hip roll": 40,
            "knee pitch": 40,
            "ankle pitch": 40,
            "shoulder pitch": 40,
            "shoulder yaw": 40,
            "shoulder roll": 40,
            "elbow pitch": 40,
            "wrist roll": 40,
        }

    @classmethod
    def friction(cls) -> Dict[str, float]:
        return {
            "hip pitch": 0.0,
            "hip yaw": 0.0,
            "hip roll": 0.0,
            "knee pitch": 0.0,
            "ankle pitch": 0.0,
            "ankle roll": 0.1,
        }


def print_joints() -> None:
    joints = Robot.all_joints()
    assert len(joints) == len(set(joints)), "Duplicate joint names found!"
    print(Robot())


if __name__ == "__main__":
    # python -m sim.Robot.joints
    print_joints()
