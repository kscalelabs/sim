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
#     wrist_roll = "wrist roll"
#     wrist_pitch = "wrist pitch"
#     wrist_yaw = "wrist yaw"
#     left_finger = "left hand left finger"
#     right_finger = "left hand right finger"


# class LeftArm(Node):
#     shoulder_yaw = "shoulder yaw"
#     shoulder_pitch = "shoulder pitch"
#     shoulder_roll = "shoulder roll"
#     elbow_pitch = "elbow pitch"
#     hand = LeftHand()


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


class Stompy(Node):
    left_arm = LeftArm()
    right_arm = RightArm()
    legs = Legs()

    @classmethod
    def default_positions(cls) -> Dict[str, float]:
        return {}

    @classmethod
    def default_standing(cls) -> Dict[str, float]:
        return {
            # arms
            Stompy.left_arm.shoulder_pitch: -1.450,
            Stompy.left_arm.shoulder_yaw: 0.023,
            Stompy.left_arm.shoulder_roll: 0.096,
            Stompy.right_arm.shoulder_pitch: 1.546,
            Stompy.right_arm.shoulder_yaw: 0.032,
            Stompy.right_arm.shoulder_roll: -0.0644,
            Stompy.left_arm.elbow_pitch: -2.1911,
            Stompy.right_arm.elbow_pitch: -2.19,
            # hands
            Stompy.left_arm.hand.wrist_roll: 0,
            Stompy.right_arm.hand.wrist_roll: 0,
            # legs
            Stompy.legs.left.hip_pitch: 3.0,
            Stompy.legs.left.hip_roll: 1.9077,
            Stompy.legs.left.hip_yaw: -0.0636,
            Stompy.legs.left.knee_pitch: 0.0644,
            Stompy.legs.left.ankle_pitch: -3.0,
            # right leg
            Stompy.legs.right.hip_pitch: -0.2790,
            Stompy.legs.right.hip_roll: 1.6738,
            Stompy.legs.right.hip_yaw: 2.9990,
            Stompy.legs.right.knee_pitch: -0.0106,
            Stompy.legs.right.ankle_pitch: 0.0568,
        }

    @classmethod
    def default_limits(cls) -> Dict[str, Dict[str, float]]:
        return {
            # arms
            Stompy.left_arm.shoulder_pitch: {
                "lower": -2.2354,  # -1.450 - 0.7854
                "upper": -0.6646,  # -1.450 + 0.7854
            },
            Stompy.left_arm.shoulder_yaw: {
                "lower": -0.7624,  # 0.023 - 0.7854
                "upper": 0.8084,  # 0.023 + 0.7854
            },
            Stompy.left_arm.shoulder_roll: {
                "lower": -0.6894,  # 0.096 - 0.7854
                "upper": 0.8814,  # 0.096 + 0.7854
            },
            Stompy.right_arm.shoulder_pitch: {
                "lower": 0.7606,  # 1.546 - 0.7854
                "upper": 2.3314,  # 1.546 + 0.7854
            },
            Stompy.right_arm.shoulder_yaw: {
                "lower": -0.7534,  # 0.032 - 0.7854
                "upper": 0.8174,  # 0.032 + 0.7854
            },
            Stompy.right_arm.shoulder_roll: {
                "lower": -0.8498,  # -0.0644 - 0.7854
                "upper": 0.7210,  # -0.0644 + 0.7854
            },
            Stompy.left_arm.elbow_pitch: {
                "lower": -2.9765,  # -2.1911 - 0.7854
                "upper": -1.4057,  # -2.1911 + 0.7854
            },
            Stompy.right_arm.elbow_pitch: {
                "lower": -2.9754,  # -2.19 - 0.7854
                "upper": -1.4046,  # -2.19 + 0.7854
            },
            # legs
            Stompy.legs.left.hip_pitch: {
                "lower": 2.2146,  # 3.0 - 0.7854
                "upper": 3.7854,  # 3.0 + 0.7854
            },
            Stompy.legs.left.hip_roll: {
                "lower": 1.1223,  # 1.9077 - 0.7854
                "upper": 2.6931,  # 1.9077 + 0.7854
            },
            Stompy.legs.left.hip_yaw: {
                "lower": -0.8490,  # -0.0636 - 0.7854
                "upper": 0.7218,  # -0.0636 + 0.7854
            },
            Stompy.legs.left.knee_pitch: {
                "lower": -0.7210,  # 0.0644 - 0.7854
                "upper": 0.8498,  # 0.0644 + 0.7854
            },
            Stompy.legs.left.ankle_pitch: {
                "lower": -3.7854,  # -3.0 - 0.7854
                "upper": -2.2146,  # -3.0 + 0.7854
            },
            # right leg
            Stompy.legs.right.hip_pitch: {
                "lower": -1.0644,  # -0.2790 - 0.7854
                "upper": 0.5064,  # -0.2790 + 0.7854
            },
            Stompy.legs.right.hip_roll: {
                "lower": 0.8884,  # 1.6738 - 0.7854
                "upper": 2.4592,  # 1.6738 + 0.7854
            },
            Stompy.legs.right.hip_yaw: {
                "lower": 2.2136,  # 2.9990 - 0.7854
                "upper": 3.7844,  # 2.9990 + 0.7854
            },
            Stompy.legs.right.knee_pitch: {
                "lower": -0.7960,  # -0.0106 - 0.7854
                "upper": 0.7748,  # -0.0106 + 0.7854
            },
            Stompy.legs.right.ankle_pitch: {
                "lower": -0.7286,  # 0.0568 - 0.7854
                "upper": 0.8422,  # 0.0568 + 0.7854
            },
        }


class StompyFixed(Stompy):
    left_arm = LeftArm()
    right_arm = RightArm()
    legs = Legs()

    @classmethod
    def default_standing(cls) -> Dict[str, float]:
        return {
            # arms
            Stompy.left_arm.shoulder_pitch: -0.534,
            Stompy.left_arm.shoulder_yaw: 2.54,
            Stompy.left_arm.shoulder_roll: -0.0314,
            Stompy.right_arm.shoulder_pitch: 2.45,
            Stompy.right_arm.shoulder_yaw: 3.77,
            Stompy.right_arm.shoulder_roll: -0.0314,
            Stompy.left_arm.elbow_pitch: 2.35,
            Stompy.right_arm.elbow_pitch: 2.65,
            # hands
            Stompy.left_arm.hand.wrist_roll: 1.79,
            Stompy.right_arm.hand.wrist_roll: -2.13,
            # legs
            Stompy.legs.left.hip_pitch: -1.6,
            Stompy.legs.left.hip_roll: 1.41,
            Stompy.legs.left.hip_yaw: -2.12,
            Stompy.legs.left.knee_pitch: 2.01,
            Stompy.legs.left.ankle_pitch: 0.238,
            Stompy.legs.right.hip_pitch: 1.76,
            Stompy.legs.right.hip_roll: -1.54,
            Stompy.legs.right.hip_yaw: 0.967,
            Stompy.legs.right.knee_pitch: 2.07,
            Stompy.legs.right.ankle_pitch: 0.377,
        }

    @classmethod
    def default_limits(cls) -> Dict[str, Dict[str, float]]:
        return {
            Stompy.left_arm.shoulder_pitch: {
                "lower": -0.584,
                "upper": -0.484,
            },
            Stompy.left_arm.shoulder_yaw: {
                "lower": 2.49,
                "upper": 2.59,
            },
            Stompy.left_arm.shoulder_roll: {
                "lower": -0.0814,
                "upper": 0.0186,
            },
            Stompy.right_arm.shoulder_pitch: {
                "lower": 2.40,
                "upper": 2.50,
            },
            Stompy.right_arm.shoulder_yaw: {
                "lower": 3.72,
                "upper": 3.82,
            },
            Stompy.right_arm.shoulder_roll: {
                "lower": -0.0814,
                "upper": 0.0186,
            },
            Stompy.left_arm.elbow_pitch: {
                "lower": 2.30,
                "upper": 2.40,
            },
            Stompy.right_arm.elbow_pitch: {
                "lower": 2.60,
                "upper": 2.70,
            },
            Stompy.left_arm.hand.wrist_roll: {
                "lower": 1.74,
                "upper": 1.84,
            },
            Stompy.right_arm.hand.wrist_roll: {
                "lower": -2.18,
                "upper": -2.08,
            },
            Stompy.legs.left.hip_pitch: {
                "lower": -1.65,
                "upper": -1.55,
            },
            Stompy.legs.left.hip_roll: {
                "lower": 1.36,
                "upper": 1.46,
            },
            Stompy.legs.left.hip_yaw: {
                "lower": -2.17,
                "upper": -2.07,
            },
            Stompy.legs.left.knee_pitch: {
                "lower": 1.96,
                "upper": 2.06,
            },
            Stompy.legs.left.ankle_pitch: {
                "lower": 0.188,
                "upper": 0.288,
            },
            Stompy.legs.right.hip_pitch: {
                "lower": 1.71,
                "upper": 1.81,
            },
            Stompy.legs.right.hip_roll: {
                "lower": -1.59,
                "upper": -1.49,
            },
            Stompy.legs.right.hip_yaw: {
                "lower": 0.917,
                "upper": 1.017,
            },
            Stompy.legs.right.knee_pitch: {
                "lower": 2.02,
                "upper": 2.12,
            },
            Stompy.legs.right.ankle_pitch: {
                "lower": 0.327,
                "upper": 0.427,
            },
        }


def print_joints() -> None:
    joints = Stompy.all_joints()
    assert len(joints) == len(set(joints)), "Duplicate joint names found!"
    print(Stompy())


if __name__ == "__main__":
    # python -m sim.stompy.joints
    print_joints()
