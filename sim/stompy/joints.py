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


class Torso(Node):
    roll = "torso roll"


class LeftHand(Node):
    wrist_roll = "left wrist roll"
    wrist_pitch = "left wrist pitch"
    wrist_yaw = "left wrist yaw"


class LeftArm(Node):
    shoulder_yaw = "left shoulder yaw"
    shoulder_pitch = "left shoulder pitch"
    shoulder_roll = "left shoulder roll"
    elbow_pitch = "left elbow pitch"
    hand = LeftHand()


class RightHand(Node):
    wrist_roll = "right wrist roll"
    wrist_pitch = "right wrist pitch"
    wrist_yaw = "right wrist yaw"


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
    def default_standing(cls) -> Dict[str, float]:
        return {
            Stompy.torso.roll: -0.502,
            # arms
            Stompy.left_arm.shoulder_pitch: -0.251,
            Stompy.left_arm.shoulder_yaw: 1.82,
            Stompy.left_arm.shoulder_roll: -1.44,
            Stompy.right_arm.shoulder_pitch: 2.7,
            Stompy.right_arm.shoulder_yaw: -1.82,
            Stompy.right_arm.shoulder_roll: -2.57,
            Stompy.left_arm.elbow_pitch: 2.07,
            Stompy.right_arm.elbow_pitch: -2.57,
            # hands
            Stompy.left_arm.hand.wrist_roll: -2.51,
            Stompy.left_arm.hand.wrist_pitch: 3.33,
            Stompy.left_arm.hand.wrist_yaw: 0.0628,
            Stompy.right_arm.hand.wrist_roll: 0,
            Stompy.right_arm.hand.wrist_pitch: 0.251,
            Stompy.right_arm.hand.wrist_yaw: 1.38,
            # legs
            Stompy.legs.left.hip_pitch: 0.502,
            Stompy.legs.left.hip_roll: -1.57,
            Stompy.legs.left.hip_yaw: -2.07,
            Stompy.legs.left.knee_pitch: 3.39,
            Stompy.legs.left.ankle_pitch: 1,
            Stompy.legs.left.ankle_roll: 1.76,
            Stompy.legs.right.hip_pitch: 1.13,
            Stompy.legs.right.hip_roll: 0,
            Stompy.legs.right.hip_yaw: 1.07,
            Stompy.legs.right.knee_pitch: 0.879,
            Stompy.legs.right.ankle_pitch: -0.502,
            Stompy.legs.right.ankle_roll: 1.76,
        }

    @classmethod
    def default_limits(cls) -> Dict[str, Dict[str, float]]:
        return {
            # torso
            Stompy.torso.roll: {
                "lower": -0.502,
                "upper": -0.501,
            },
            # left arm
            Stompy.left_arm.shoulder_pitch: {
                "lower": -0.251,
                "upper": -0.250,
            },
            Stompy.left_arm.shoulder_yaw: {
                "lower": 1.82,
                "upper": 1.821,
            },
            Stompy.left_arm.shoulder_roll: {
                "lower": -1.44,
                "upper": -1.439,
            },
            Stompy.left_arm.elbow_pitch: {
                "lower": 2.07,
                "upper": 2.071,
            },
            Stompy.left_arm.hand.wrist_roll: {
                "lower": -2.51,
                "upper": -2.509,
            },
            Stompy.left_arm.hand.wrist_pitch: {
                "lower": 3.33,
                "upper": 3.331,
            },
            Stompy.left_arm.hand.wrist_yaw: {
                "lower": 0.0628,
                "upper": 0.0638,
            },
            # right arm
            Stompy.right_arm.shoulder_pitch: {
                "lower": 2.7,
                "upper": 2.701,
            },
            Stompy.right_arm.shoulder_yaw: {
                "lower": -1.82,
                "upper": -1.819,
            },
            Stompy.right_arm.shoulder_roll: {
                "lower": -2.57,
                "upper": -2.569,
            },
            Stompy.right_arm.elbow_pitch: {
                "lower": -2.57,
                "upper": -2.569,
            },
            Stompy.right_arm.hand.wrist_roll: {
                "lower": 0,
                "upper": 0.001,
            },
            Stompy.right_arm.hand.wrist_pitch: {
                "lower": 0.251,
                "upper": 0.252,
            },
            Stompy.right_arm.hand.wrist_yaw: {
                "lower": 1.38,
                "upper": 1.381,
            },
            # left leg
            Stompy.legs.left.hip_pitch: {
                "lower": -0.498,
                "upper": 1.502,
            },
            Stompy.legs.left.hip_roll: {
                "lower": -2.57,
                "upper": -0.57,
            },
            Stompy.legs.left.hip_yaw: {
                "lower": -3.07,
                "upper": -1.07,
            },
            Stompy.legs.left.knee_pitch: {
                "lower": 2.39,
                "upper": 4.39,
            },
            Stompy.legs.left.ankle_pitch: {
                "lower": 0,
                "upper": 2,
            },
            Stompy.legs.left.ankle_roll: {
                "lower": 0.76,
                "upper": 2.76,
            },
            Stompy.legs.right.hip_pitch: {
                "lower": 0.13,
                "upper": 2.13,
            },
            Stompy.legs.right.hip_roll: {
                "lower": -1,
                "upper": 1,
            },
            Stompy.legs.right.hip_yaw: {
                "lower": 0.07,
                "upper": 2.07,
            },
            Stompy.legs.right.knee_pitch: {
                "lower": -0.121,
                "upper": 1.879,
            },
            Stompy.legs.right.ankle_pitch: {
                "lower": -1.502,
                "upper": 0.498,
            },
            Stompy.legs.right.ankle_roll: {
                "lower": 0.76,
                "upper": 2.76,
            },
        }

    # p_gains
    @classmethod
    def stiffness(cls) -> Dict[str, float]:
        return {
            "hip pitch": 250,
            "hip yaw": 150,
            "hip roll": 150,
            "knee pitch": 150,
            "ankle pitch": 45,
            "ankle roll": 45,
            "shoulder pitch": 0,
            "shoulder yaw": 0,
            "shoulder roll": 0,
            "elbow pitch": 0,
            "wrist roll": 0,
            "wrist pitch": 0,
            "wrist yaw": 0,
            "torso roll": 0,
        }

    # d_gains
    @classmethod
    def damping(cls) -> Dict[str, float]:
        return {
            "hip pitch": 15,
            "hip yaw": 10,
            "hip roll": 10,
            "knee pitch": 10,
            "ankle pitch": 10,
            "ankle roll": 10,
            "shoulder pitch": 0,
            "shoulder yaw": 0,
            "shoulder roll": 0,
            "elbow pitch": 0,
            "wrist roll": 0,
            "wrist pitch": 0,
            "wrist yaw": 0,
            "torso roll": 0,
        }

    # pos_limits
    @classmethod
    def effort(cls) -> Dict[str, float]:
        return {
            "hip pitch": 150,
            "hip yaw": 90,
            "hip roll": 90,
            "knee pitch": 90,
            "ankle pitch": 24,
            "ankle roll": 24,
            "shoulder pitch": 0,
            "shoulder yaw": 0,
            "shoulder roll": 0,
            "elbow pitch": 0,
            "wrist roll": 0,
            "wrist pitch": 0,
            "wrist yaw": 0,
            "torso roll": 0,
        }

    # vel_limits
    @classmethod
    def velocity(cls) -> Dict[str, float]:
        return {
            "hip pitch": 40,
            "hip yaw": 40,
            "hip roll": 40,
            "knee pitch": 40,
            "ankle pitch": 12,
            "ankle roll": 12,
            "shoulder pitch": 0,
            "shoulder yaw": 0,
            "shoulder roll": 0,
            "elbow pitch": 0,
            "wrist roll": 0,
            "wrist pitch": 0,
            "wrist yaw": 0,
            "torso roll": 0,
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
    joints = Stompy.all_joints()
    assert len(joints) == len(set(joints)), "Duplicate joint names found!"
    print(Stompy())


if __name__ == "__main__":
    # python -m sim.stompy.joints
    print_joints()
