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


# class LeftHand(Node):
#     wrist_roll = "left hand roll"
#     gripper = "left hand gripper"


class LeftArm(Node):
    shoulder_yaw = "left_shoulder_yaw"
    shoulder_pitch = "left_shoulder_pitch"
    # shoulder_roll = "left_shoulder_roll"
    elbow_pitch = "left_elbow_yaw" # FIXME: yaw vs pitch
    # hand = LeftHand()


# class RightHand(Node):
#     wrist_roll = "right hand roll"
#     gripper = "right hand gripper"


class RightArm(Node):
    shoulder_yaw = "right_shoulder_yaw"
    shoulder_pitch = "right_shoulder_pitch"
    # shoulder_roll = "right_shoulder_roll"
    elbow_pitch = "right_elbow_yaw"  # FIXME: yaw vs pitch
    # hand = RightHand()


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
    rotation = [0.0, 0.0, 0, 1]

    left_arm = LeftArm()
    right_arm = RightArm()
    legs = Legs()

    @classmethod
    def default_standing(cls) -> Dict[str, float]:
        return {
            # arms
            Robot.left_arm.shoulder_pitch: 2.25,
            Robot.left_arm.shoulder_yaw: 1.57,
            # Robot.left_arm.shoulder_roll: 3.14,
            Robot.left_arm.elbow_pitch: -1.61,
            # Robot.left_arm.hand.wrist_roll: -1.56,
            # Robot.left_arm.hand.gripper: -0.1,
            Robot.right_arm.shoulder_pitch: 3.45,
            Robot.right_arm.shoulder_yaw: -0.1,
            # Robot.right_arm.shoulder_roll: -1.61,
            Robot.right_arm.elbow_pitch: -1.49,
            # Robot.right_arm.hand.wrist_roll: 0,
            # Robot.right_arm.hand.gripper: 0.1,
            # legs
            Robot.legs.left.hip_pitch: 0.33,
            Robot.legs.left.hip_roll: -1.52,
            Robot.legs.left.hip_yaw: 4.67,
            Robot.legs.left.knee_pitch: -0.61,
            Robot.legs.left.ankle_pitch: 1.88,
            Robot.legs.right.hip_pitch: 2.91,
            Robot.legs.right.hip_roll: 3.24,
            Robot.legs.right.hip_yaw: 3.22,
            Robot.legs.right.knee_pitch: 0.65,
            Robot.legs.right.ankle_pitch: -0.54,
        }

    @classmethod
    def default_limits2(cls) -> Dict[str, Dict[str, float]]:
        return {
            # left arm
            Robot.left_arm.shoulder_pitch: {
                "lower": 2.54,
                "upper": 2.56,
            },
            Robot.left_arm.shoulder_yaw: {
                "lower": 1.56,
                "upper": 1.58,
            },
            # Robot.left_arm.shoulder_roll: {
            #     "lower": 3.13,
            #     "upper": 3.14,
            # },
            Robot.left_arm.elbow_pitch: {
                "lower": -1.56,
                "upper": -1.58,
            },
            # Robot.left_arm.hand.wrist_roll: {
            #     "lower": -1.56,
            #     "upper": -1.58,
            # },
            # Robot.left_arm.hand.gripper: {
            #     "lower": 0,
            #     "upper": 1.57,
            # },
            # right arm
            Robot.right_arm.shoulder_pitch: {
                "lower": 3.119,
                "upper": 3.121,
            },
            Robot.right_arm.shoulder_yaw: {
                "lower": 1.981,
                "upper": 1.979,
            },
            # Robot.right_arm.shoulder_roll: {
            #     "lower": -1.381,
            #     "upper": -1.979,
            # },
            Robot.right_arm.elbow_pitch: {
                "lower": -3.319,
                "upper": 3.321,
            },
            # Robot.right_arm.hand.wrist_roll: {
            #     "lower": -0.001,
            #     "upper": 0.001,
            # },
            # Robot.right_arm.hand.gripper: {
            #     "lower": 0,
            #     "upper": 1.57,
            # },
            # left leg
            Robot.legs.left.hip_pitch: {
                "lower": -1.14,
                "upper": 1.14,
            },
            Robot.legs.left.hip_roll: {
                "lower": -3.5,
                "upper": 0.5,
            },
            Robot.legs.left.hip_yaw: {
                "lower": 3.14,
                "upper": 5.14,
            },
            Robot.legs.left.knee_pitch: {
                "lower": -2,
                "upper": 0,
            },
            Robot.legs.left.ankle_pitch: {
                "lower": 1.4,
                "upper": 2.2,
            },
            # right leg
            Robot.legs.right.hip_pitch: {
                "lower": 0.55,
                "upper": 3.55,
            },
            Robot.legs.right.hip_roll: {
                "lower": 2.75,
                "upper": 3.99,
            },
            Robot.legs.right.hip_yaw: {
                "lower": 2.24,
                "upper": 4.24,
            },
            Robot.legs.right.knee_pitch: {
                "lower": 0,
                "upper": 2,
            },
            Robot.legs.right.ankle_pitch: {
                "lower": -1.0,
                "upper": 0.2,
            },
        }

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
            # Robot.left_arm.shoulder_roll: {
            #     "lower": 2.63,
            #     "upper": 3.64,
            # },
            Robot.left_arm.elbow_pitch: {
                "lower": -2.06,
                "upper": -1.08,
            },
            # Robot.left_arm.hand.wrist_roll: {
            #     "lower": -2.06,
            #     "upper": -1.08,
            # },
            # Robot.left_arm.hand.gripper: {
            #     "lower": -0.5,
            #     "upper": 2.07,
            # },
            # right arm
            Robot.right_arm.shoulder_pitch: {
                "lower": 2.619,
                "upper": 3.621,
            },
            Robot.right_arm.shoulder_yaw: {
                "lower": -1.481,
                "upper": 1,
            },
            # Robot.right_arm.shoulder_roll: {
            #     "lower": -1.881,
            #     "upper": -1.479,
            # },
            Robot.right_arm.elbow_pitch: {
                "lower": -3.819,
                "upper": 3.821,
            },
            # Robot.right_arm.hand.wrist_roll: {
            #     "lower": -0.501,
            #     "upper": 0.501,
            # },
            # Robot.right_arm.hand.gripper: {
            #     "lower": -0.5,
            #     "upper": 2.07,
            # },
            # left leg
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
                "lower": 0.9,
                "upper": 2.7,
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
                "lower": -1.5,
                "upper": 0.7,
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
            "hand roll": 45,
            "gripper": 45,
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
            "hand roll": 5,
            "gripper": 5,
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
            "hand roll": 17,
            "gripper": 17,
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
            "hand roll": 40,
            "gripper": 40,
        }

    @classmethod
    def friction(cls) -> Dict[str, float]:
        return {
            "hip pitch": 0.0,
            "hip yaw": 0.0,
            "hip roll": 0.0,
            "knee pitch": 0.0,
            "ankle pitch": 0.0,
            "hand roll": 0.0,
            "gripper": 0.0,
        }


def print_joints() -> None:
    joints = Robot.all_joints()
    assert len(joints) == len(set(joints)), "Duplicate joint names found!"
    print(Robot())


if __name__ == "__main__":
    # python -m sim.Robot.joints
    print_joints()
