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
            # left arm
            Stompy.left_arm.shoulder_pitch: -0.502,
            Stompy.left_arm.shoulder_yaw: 1.26,
            Stompy.left_arm.shoulder_roll: -3.01,
            Stompy.left_arm.elbow_pitch: 4.46,
            Stompy.left_arm.hand.wrist_roll: -1.57,
            Stompy.left_arm.hand.wrist_pitch: 0,
            Stompy.left_arm.hand.wrist_yaw: 0.0628,
            # right arm
            Stompy.right_arm.shoulder_pitch: 2.95,
            Stompy.right_arm.shoulder_yaw: -1.26,
            Stompy.right_arm.shoulder_roll: -0.126,
            Stompy.right_arm.elbow_pitch: 1.13,
            Stompy.right_arm.hand.wrist_roll: -1.76,
            Stompy.right_arm.hand.wrist_pitch: 2.95,
            Stompy.right_arm.hand.wrist_yaw: 0.251,
            # legs
            Stompy.torso.roll: -0.502,
            Stompy.legs.right.hip_pitch: -0.988,
            Stompy.legs.right.hip_yaw: 1.07,
            Stompy.legs.right.hip_roll: 0,
            Stompy.legs.right.knee_pitch: 0.879,
            Stompy.legs.right.ankle_pitch: 0.358,
            Stompy.legs.left.ankle_roll: 1.76,
            Stompy.legs.left.hip_pitch: 0.502,
            Stompy.legs.left.hip_yaw: -2.07,
            Stompy.legs.left.hip_roll: -1.57,
            Stompy.legs.left.knee_pitch: 2.99,
            Stompy.legs.left.ankle_pitch: 1,
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
                "lower": -0.502,
                "upper": -0.501,
            },
            Stompy.left_arm.shoulder_yaw: {
                "lower": 1.26,
                "upper": 1.261,
            },
            Stompy.left_arm.shoulder_roll: {
                "lower": -3.01,
                "upper": -3.009,
            },
            Stompy.left_arm.elbow_pitch: {
                "lower": 4.46,
                "upper": 4.461,
            },
            Stompy.left_arm.hand.wrist_roll: {
                "lower": -1.57,
                "upper": -1.569,
            },
            Stompy.left_arm.hand.wrist_pitch: {
                "lower": 0,
                "upper": 0.001,
            },
            Stompy.left_arm.hand.wrist_yaw: {
                "lower": 0.0628,
                "upper": 0.0638,
            },
            # right arm
            Stompy.right_arm.shoulder_pitch: {
                "lower": 2.95,
                "upper": 2.951,
            },
            Stompy.right_arm.shoulder_yaw: {
                "lower": -1.26,
                "upper": -1.259,
            },
            Stompy.right_arm.shoulder_roll: {
                "lower": -0.126,
                "upper": -0.125,
            },
            Stompy.right_arm.elbow_pitch: {
                "lower": 1.13,
                "upper": 1.131,
            },
            Stompy.right_arm.hand.wrist_roll: {
                "lower": -1.76,
                "upper": -1.759,
            },
            Stompy.right_arm.hand.wrist_pitch: {
                "lower": 2.95,
                "upper": 2.951,
            },
            Stompy.right_arm.hand.wrist_yaw: {
                "lower": 0.251,
                "upper": 0.252,
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
                "lower": 1.99,
                "upper": 3.99,
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
                "lower": -1.988,
                "upper": 0.012,
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
                "lower": -0.462,
                "upper": 1.358,
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
            "left shoulder pitch": 1,
            "left shoulder yaw": 1,
            "left shoulder roll": 1,
            "left elbow pitch": 1,
            "left wrist roll": 1,
            "left wrist pitch": 1,
            "left wrist yaw": 1,
            "right shoulder pitch": 1,
            "right shoulder yaw": 1,
            "right shoulder roll": 1,
            "right elbow pitch": 1,
            "right wrist roll": 1,
            "right wrist pitch": 1,
            "right wrist yaw": 1,
            "torso roll": 1,
            "right hip pitch": 150,
            "right hip yaw": 90,
            "right hip roll": 90,
            "right knee pitch": 90,
            "right ankle pitch": 45,
            "left ankle roll": 45,
            "left hip pitch": 150,
            "left hip yaw": 90,
            "left hip roll": 90,
            "left knee pitch": 90,
            "left ankle pitch": 45,
            "right ankle roll": 45,
        }

    # d_gains
    @classmethod
    def damping(cls) -> Dict[str, float]:
        return {
            "left shoulder pitch": 1,
            "left shoulder yaw": 1,
            "left shoulder roll": 1,
            "left elbow pitch": 1,
            "left wrist roll": 1,
            "left wrist pitch": 1,
            "left wrist yaw": 1,
            "right shoulder pitch": 1,
            "right shoulder yaw": 1,
            "right shoulder roll": 1,
            "right elbow pitch": 1,
            "right wrist roll": 1,
            "right wrist pitch": 1,
            "right wrist yaw": 1,
            "torso roll": 1,
            "right hip pitch": 15,
            "right hip yaw": 10,
            "right hip roll": 10,
            "right knee pitch": 10,
            "right ankle pitch": 10,
            "left ankle roll": 10,
            "left hip pitch": 15,
            "left hip yaw": 10,
            "left hip roll": 10,
            "left knee pitch": 10,
            "left ankle pitch": 10,
            "right ankle roll": 10,
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
            "shoulder pitch": 1,
            "shoulder yaw": 1,
            "shoulder roll": 1,
            "elbow pitch": 1,
            "wrist roll": 1,
            "wrist pitch": 1,
            "wrist yaw": 1,
            "torso roll": 1,
        }

    # vel_limits
    @classmethod
    def velocity(cls) -> Dict[str, float]:
        return {
            "hip pitch": 40,
            "hip yaw": 40,
            "hip roll": 40,
            "knee pitch": 40,
            "ankle pitch": 24,
            "ankle roll": 24,
            "shoulder yaw": 40,
            "shoulder roll": 40,
            "elbow pitch": 40,
            "wrist roll": 40,
            "wrist pitch": 40,
            "wrist yaw": 40,
            "torso roll": 40,
        }

    @classmethod
    def friction(cls) -> Dict[str, float]:
        return {
            "hip pitch": 0.0,
            "hip yaw": 0.0,
            "hip roll": 0.0,
            "knee pitch": 0.0,
            "ankle pitch": 0.1,
            "ankle roll": 0.1,
        }


def print_joints() -> None:
    joints = Stompy.all_joints()
    assert len(joints) == len(set(joints)), "Duplicate joint names found!"
    print(Stompy())


if __name__ == "__main__":
    # python -m sim.stompy.joints
    print_joints()
