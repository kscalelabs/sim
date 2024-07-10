"""Defines a more Pythonic interface for specifying the joint names.

The best way to re-generate this snippet for a new robot is to use the
`sim/scripts/print_joints.py` script. This script will print out a hierarchical
tree of the various joint names in the robot.
"""

import textwrap
from abc import ABC
from typing import Dict, List, Tuple, Union

import numpy as np


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


class Torso(Node):
    roll = "torso roll"


class LeftHand(Node):
    wrist_roll = "wrist roll"
    wrist_pitch = "wrist pitch"
    wrist_yaw = "wrist yaw"
    left_finger = "left hand left finger"
    right_finger = "left hand right finger"


class LeftArm(Node):
    shoulder_yaw = "shoulder yaw"
    shoulder_pitch = "shoulder pitch"
    shoulder_roll = "shoulder roll"
    elbow_pitch = "elbow pitch"
    hand = LeftHand()


class RightHand(Node):
    wrist_roll = "right wrist roll"
    wrist_pitch = "right wrist pitch"
    wrist_yaw = "right wrist yaw"
    left_finger = "right hand left finger"
    right_finger = "right hand right finger"


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
    torso = Torso()
    left_arm = LeftArm()
    right_arm = RightArm()
    legs = Legs()

    @classmethod
    def default_positions(cls) -> Dict[str, float]:
        return {
            Stompy.torso.roll: 0.0,
            Stompy.left_arm.shoulder_yaw: np.deg2rad(60),
            Stompy.left_arm.shoulder_pitch: np.deg2rad(60),
            Stompy.right_arm.shoulder_yaw: np.deg2rad(-60),
        }

    @classmethod
    def default_standing(cls) -> Dict[str, float]:
        return {
            Stompy.torso.roll: 2.58,
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
            Stompy.left_arm.hand.left_finger: 0.0,
            Stompy.left_arm.hand.right_finger: 0.0,
            Stompy.right_arm.hand.left_finger: 0.0,
            Stompy.right_arm.hand.right_finger: 0.0,
            Stompy.left_arm.hand.wrist_roll: 1.79,
            Stompy.left_arm.hand.wrist_pitch: 1.35,
            Stompy.left_arm.hand.wrist_yaw: 1.07,
            Stompy.right_arm.hand.wrist_roll: -2.13,
            Stompy.right_arm.hand.wrist_pitch: 1.79,
            Stompy.right_arm.hand.wrist_yaw: -0.251,
            # legs
            Stompy.legs.left.hip_pitch: -1.6,
            Stompy.legs.left.hip_roll: 1.41,
            Stompy.legs.left.hip_yaw: -2.12,
            Stompy.legs.left.knee_pitch: 2.01,
            Stompy.legs.left.ankle_pitch: 0.238,
            Stompy.legs.left.ankle_roll: 1.85,
            Stompy.legs.right.hip_pitch: 1.76,
            Stompy.legs.right.hip_roll: -1.54,
            Stompy.legs.right.hip_yaw: 0.967,
            Stompy.legs.right.knee_pitch: 2.07,
            Stompy.legs.right.ankle_pitch: 0.377,
            Stompy.legs.right.ankle_roll: 1.92,
        }

    @classmethod
    def default_sitting(cls) -> Dict[str, float]:
        return {
            Stompy.torso.roll: 0.0,
            # arms
            Stompy.left_arm.shoulder_pitch: -0.126,
            Stompy.left_arm.shoulder_yaw: 2.12,
            Stompy.left_arm.shoulder_roll: 1.89,
            Stompy.right_arm.shoulder_pitch: -1.13,
            Stompy.right_arm.shoulder_yaw: 2.1,
            Stompy.right_arm.shoulder_roll: -1.23,
            Stompy.left_arm.elbow_pitch: 3.0,
            Stompy.right_arm.elbow_pitch: 3.0,
            # hands
            Stompy.left_arm.hand.left_finger: 0.0,
            Stompy.left_arm.hand.right_finger: 0.0,
            Stompy.right_arm.hand.left_finger: 0.0,
            Stompy.right_arm.hand.right_finger: 0.0,
            Stompy.left_arm.hand.wrist_roll: -0.346,
            Stompy.left_arm.hand.wrist_pitch: -0.251,
            Stompy.left_arm.hand.wrist_yaw: 0.377,
            Stompy.right_arm.hand.wrist_roll: -3.14,
            Stompy.right_arm.hand.wrist_pitch: -0.437,
            Stompy.right_arm.hand.wrist_yaw: 0.911,
            # legs
            Stompy.legs.left.hip_pitch: -1.55,
            Stompy.legs.left.hip_roll: 1.46,
            Stompy.legs.left.hip_yaw: 1.45,
            Stompy.legs.left.knee_pitch: 2.17,
            Stompy.legs.left.ankle_pitch: 0.238,
            Stompy.legs.left.ankle_roll: 1.79,
            Stompy.legs.right.hip_pitch: -1.55,
            Stompy.legs.right.hip_roll: -1.67,
            Stompy.legs.right.hip_yaw: 1.04,
            Stompy.legs.right.knee_pitch: 2.01,
            Stompy.legs.right.ankle_pitch: 0.44,
            Stompy.legs.right.ankle_roll: 1.79,
        }

    @classmethod
    def default_limits(cls) -> Dict[str, Dict[str, float]]:
        return {
            Stompy.torso.roll: {
                "lower": -4.36332,
                "upper": 4.36332,
            },
            Stompy.left_arm.shoulder_pitch: {
                "lower": -0.0,
                "upper": 0.2,
            },
            Stompy.left_arm.shoulder_yaw: {
                "lower": 0.97738438,
                "upper": 5.3058009,
            },
            Stompy.left_arm.shoulder_roll: {
                "lower": -4.71239,
                "upper": 4.71239,
            },
            Stompy.right_arm.shoulder_pitch: {
                "lower": -4.71239,
                "upper": 4.71239,
            },
            Stompy.right_arm.shoulder_yaw: {
                "lower": 0.97738438,
                "upper": 5.3058009,
            },
            Stompy.right_arm.shoulder_roll: {
                "lower": -4.71239,
                "upper": 4.71239,
            },
            Stompy.left_arm.hand.wrist_roll: {
                "lower": -4.71239,
                "upper": 4.71239,
            },
            Stompy.left_arm.hand.wrist_pitch: {
                "lower": -3.66519,
                "upper": -1.39626,
            },
            Stompy.left_arm.hand.wrist_yaw: {
                "lower": 0,
                "upper": 1.5708,
            },
            Stompy.right_arm.hand.wrist_roll: {
                "lower": -4.71239,
                "upper": 4.71239,
            },
            Stompy.right_arm.hand.wrist_pitch: {
                "lower": -1.5708,
                "upper": 0.523599,
            },
            Stompy.right_arm.hand.wrist_yaw: {
                "lower": -1.5708,
                "upper": 0,
            },
            Stompy.legs.left.hip_pitch: {
                "lower": -4.712389,
                "upper": 4.712389,
            },
            Stompy.legs.left.hip_roll: {
                "lower": -3.14159,
                "upper": 0,
            },
            Stompy.legs.left.hip_yaw: {
                "lower": -1.0472,
                "upper": 2.0944,
            },
            Stompy.legs.left.knee_pitch: {
                "lower": -4.18879,
                "upper": 0,
            },
            Stompy.legs.left.ankle_pitch: {
                "lower": -1.5708,
                "upper": 2.18166,
            },
            Stompy.legs.left.ankle_roll: {
                "lower": -2.26893,
                "upper": -1.22173,
            },
            Stompy.legs.right.hip_pitch: {
                "lower": -4.712389,
                "upper": 4.712389,
            },
            Stompy.legs.right.hip_roll: {
                "lower": 0,
                "upper": 3.14159,
            },
            Stompy.legs.right.hip_yaw: {
                "lower": -1.0472,
                "upper": 2.0944,
            },
            Stompy.legs.right.knee_pitch: {
                "lower": -4.18879,
                "upper": 0,
            },
            Stompy.legs.right.ankle_pitch: {
                "lower": -1.5708,
                "upper": 2.18166,
            },
            Stompy.legs.right.ankle_roll: {
                "lower": -2.26893,
                "upper": -1.22173,
            },
            Stompy.left_arm.elbow_pitch: {
                "lower": 1.4486233,
                "higher": 5.4454273,
            },
            Stompy.right_arm.elbow_pitch: {
                "lower": 1.4486233,
                "higher": 5.4454273,
            },
            Stompy.left_arm.hand.left_finger: {
                "lower": -0.051,
                "upper": 0.0,
            },
            Stompy.left_arm.hand.right_finger: {
                "lower": 0,
                "upper": 0.051,
            },
            Stompy.right_arm.hand.left_finger: {
                "lower": -0.051,
                "upper": 0.0,
            },
            Stompy.right_arm.hand.right_finger: {
                "lower": 0,
                "upper": 0.051,
            },
        }


class StompyFixed(Stompy):
    torso = Torso()
    left_arm = LeftArm()
    right_arm = RightArm()
    legs = Legs()

    @classmethod
    def default_standing(cls) -> Dict[str, float]:
        return {
            Stompy.torso.roll: 2.58,
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
            Stompy.left_arm.hand.left_finger: 0.0,
            Stompy.left_arm.hand.right_finger: 0.0,
            Stompy.right_arm.hand.left_finger: 0.0,
            Stompy.right_arm.hand.right_finger: 0.0,
            Stompy.left_arm.hand.wrist_roll: 1.79,
            Stompy.left_arm.hand.wrist_pitch: 1.35,
            Stompy.left_arm.hand.wrist_yaw: 1.07,
            Stompy.right_arm.hand.wrist_roll: -2.13,
            Stompy.right_arm.hand.wrist_pitch: 1.79,
            Stompy.right_arm.hand.wrist_yaw: -0.251,
            # legs
            Stompy.legs.left.hip_pitch: -1.6,
            Stompy.legs.left.hip_roll: 1.41,
            Stompy.legs.left.hip_yaw: -2.12,
            Stompy.legs.left.knee_pitch: 2.01,
            Stompy.legs.left.ankle_pitch: 0.238,
            Stompy.legs.left.ankle_roll: 1.85,
            Stompy.legs.right.hip_pitch: 1.76,
            Stompy.legs.right.hip_roll: -1.54,
            Stompy.legs.right.hip_yaw: 0.967,
            Stompy.legs.right.knee_pitch: 2.07,
            Stompy.legs.right.ankle_pitch: 0.377,
            Stompy.legs.right.ankle_roll: 1.92,
        }

    @classmethod
    def default_limits(cls) -> Dict[str, Dict[str, float]]:
        return {
            Stompy.torso.roll: {
                "lower": 2.53,
                "upper": 2.63,
            },
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
            Stompy.left_arm.hand.left_finger: {
                "lower": -0.05,
                "upper": 0.05,
            },
            Stompy.left_arm.hand.right_finger: {
                "lower": -0.05,
                "upper": 0.05,
            },
            Stompy.right_arm.hand.left_finger: {
                "lower": -0.05,
                "upper": 0.05,
            },
            Stompy.right_arm.hand.right_finger: {
                "lower": -0.05,
                "upper": 0.05,
            },
            Stompy.left_arm.hand.wrist_roll: {
                "lower": 1.74,
                "upper": 1.84,
            },
            Stompy.left_arm.hand.wrist_pitch: {
                "lower": 1.30,
                "upper": 1.40,
            },
            Stompy.left_arm.hand.wrist_yaw: {
                "lower": 1.02,
                "upper": 1.12,
            },
            Stompy.right_arm.hand.wrist_roll: {
                "lower": -2.18,
                "upper": -2.08,
            },
            Stompy.right_arm.hand.wrist_pitch: {
                "lower": 1.74,
                "upper": 1.84,
            },
            Stompy.right_arm.hand.wrist_yaw: {
                "lower": -0.301,
                "upper": -0.201,
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
            Stompy.legs.left.ankle_roll: {
                "lower": 1.80,
                "upper": 1.90,
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
            Stompy.legs.right.ankle_roll: {
                "lower": 1.87,
                "upper": 1.97,
            },
        }


class ImuTorso(Torso):
    pitch = "torso_1_x8_1_dof_x8"
    pelvis_tx = "pelvis_tx"
    pelvis_tz = "pelvis_tz"
    pelvis_ty = "pelvis_ty"
    tilt = "pelvis_tilt"
    list = "pelvis_list"
    rotation = "pelvis_rotation"


class MjcfStompy(Stompy):
    torso = ImuTorso()
    left_arm = LeftArm()
    right_arm = RightArm()
    legs = Legs()

    # test the model
    @classmethod
    def default_standing(cls) -> Dict[str, float]:
        return {
            Stompy.torso.roll: 2.58,
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
            Stompy.left_arm.hand.left_finger: 0.0,
            Stompy.left_arm.hand.right_finger: 0.0,
            Stompy.right_arm.hand.left_finger: 0.0,
            Stompy.right_arm.hand.right_finger: 0.0,
            Stompy.left_arm.hand.wrist_roll: 1.79,
            Stompy.left_arm.hand.wrist_pitch: 1.35,
            Stompy.left_arm.hand.wrist_yaw: 1.07,
            Stompy.right_arm.hand.wrist_roll: -2.13,
            Stompy.right_arm.hand.wrist_pitch: 1.79,
            Stompy.right_arm.hand.wrist_yaw: -0.251,
            # legs
            Stompy.legs.left.hip_pitch: -1.6,
            Stompy.legs.left.hip_roll: 1.41,
            Stompy.legs.left.hip_yaw: -2.12,
            Stompy.legs.left.knee_pitch: 2.01,
            Stompy.legs.left.ankle_pitch: 0.238,
            Stompy.legs.left.ankle_roll: 1.85,
            Stompy.legs.right.hip_pitch: 1.76,
            Stompy.legs.right.hip_roll: -1.54,
            Stompy.legs.right.hip_yaw: 0.967,
            Stompy.legs.right.knee_pitch: 2.07,
            Stompy.legs.right.ankle_pitch: 0.377,
            Stompy.legs.right.ankle_roll: 1.92,
        }

    @classmethod
    def default_limits(cls) -> Dict[str, Dict[str, float]]:
        return {
            Stompy.torso.roll: {
                "lower": 2.53,
                "upper": 2.63,
            },
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
            Stompy.left_arm.hand.left_finger: {
                "lower": -0.05,
                "upper": 0.05,
            },
            Stompy.left_arm.hand.right_finger: {
                "lower": -0.05,
                "upper": 0.05,
            },
            Stompy.right_arm.hand.left_finger: {
                "lower": -0.05,
                "upper": 0.05,
            },
            Stompy.right_arm.hand.right_finger: {
                "lower": -0.05,
                "upper": 0.05,
            },
            Stompy.left_arm.hand.wrist_roll: {
                "lower": 1.74,
                "upper": 1.84,
            },
            Stompy.left_arm.hand.wrist_pitch: {
                "lower": 1.30,
                "upper": 1.40,
            },
            Stompy.left_arm.hand.wrist_yaw: {
                "lower": 1.02,
                "upper": 1.12,
            },
            Stompy.right_arm.hand.wrist_roll: {
                "lower": -2.18,
                "upper": -2.08,
            },
            Stompy.right_arm.hand.wrist_pitch: {
                "lower": 1.74,
                "upper": 1.84,
            },
            Stompy.right_arm.hand.wrist_yaw: {
                "lower": -0.301,
                "upper": -0.201,
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
            Stompy.legs.left.ankle_roll: {
                "lower": 1.80,
                "upper": 1.90,
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
            Stompy.legs.right.ankle_roll: {
                "lower": 1.87,
                "upper": 1.97,
            },
        }


def print_joints() -> None:
    joints = Stompy.all_joints()
    assert len(joints) == len(set(joints)), "Duplicate joint names found!"
    print(Stompy())


if __name__ == "__main__":
    # python -m sim.stompy.joints
    print_joints()
