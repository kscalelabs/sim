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
                "lower": -0.503,
                "upper": -0.501,
            },
            # left arm
            Stompy.left_arm.shoulder_pitch: {
                "lower": -0.252,
                "upper": -0.250,
            },
            Stompy.left_arm.shoulder_yaw: {
                "lower": 1.819,
                "upper": 1.821,
            },
            Stompy.left_arm.shoulder_roll: {
                "lower": -1.45,
                "upper": -1.439,
            },
            Stompy.left_arm.elbow_pitch: {
                "lower": 2.06,
                "upper": 2.071,
            },
            Stompy.left_arm.hand.wrist_roll: {
                "lower": -2.512,
                "upper": -2.509,
            },
            Stompy.left_arm.hand.wrist_pitch: {
                "lower": 3.32,
                "upper": 3.331,
            },
            Stompy.left_arm.hand.wrist_yaw: {
                "lower": 0.062,
                "upper": 0.0638,
            },
            # right arm
            Stompy.right_arm.shoulder_pitch: {
                "lower": 2.69,
                "upper": 2.701,
            },
            Stompy.right_arm.shoulder_yaw: {
                "lower": -1.83,
                "upper": -1.819,
            },
            Stompy.right_arm.shoulder_roll: {
                "lower": -2.58,
                "upper": -2.569,
            },
            Stompy.right_arm.elbow_pitch: {
                "lower": -2.58,
                "upper": -2.569,
            },
            Stompy.right_arm.hand.wrist_roll: {
                "lower": -0.01,
                "upper": 0.001,
            },
            Stompy.right_arm.hand.wrist_pitch: {
                "lower": 0.250,
                "upper": 0.252,
            },
            Stompy.right_arm.hand.wrist_yaw: {
                "lower": 1.37,
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
            "shoulder pitch": 1,
            "shoulder yaw": 1,
            "shoulder roll": 1,
            "elbow pitch": 1,
            "wrist roll": 1,
            "wrist pitch": 1,
            "wrist yaw": 1,
            "torso roll": 1,
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
            "shoulder pitch": 1,
            "shoulder yaw": 1,
            "shoulder roll": 1,
            "elbow pitch": 1,
            "wrist roll": 1,
            "wrist pitch": 1,
            "wrist yaw": 1,
            "torso roll": 1,
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
            "shoulder pitch": 24,
            "shoulder yaw": 24,
            "shoulder roll": 24,
            "elbow pitch": 24,
            "wrist roll": 24,
            "wrist pitch": 24,
            "wrist yaw": 24,
            "torso roll": 150,
        }

    # vel_limits
    @classmethod
    def velocity(cls) -> Dict[str, float]:
        # return {
        #     "hip pitch": 40,
        #     "hip yaw": 40,
        #     "hip roll": 40,
        #     "knee pitch": 40,
        #     "ankle pitch": 40,
        #     "ankle roll": 40,
        #     "shoulder pitch": 40,
        #     "shoulder yaw": 40,
        #     "shoulder roll": 40,
        #     "elbow pitch": 40,
        #     "wrist roll": 40,
        #     "wrist pitch": 40,
        #     "wrist yaw": 40,
        #     "torso roll": 40,
        # }
        return {
            "hip pitch": 150,
            "hip yaw": 150,
            "hip roll": 150,
            "knee pitch": 150,
            "ankle pitch": 150,
            "ankle roll": 150,
            "shoulder pitch": 150,
            "shoulder yaw": 150,
            "shoulder roll": 150,
            "elbow pitch": 150,
            "wrist roll": 150,
            "wrist pitch": 150,
            "wrist yaw": 150,
            "torso roll": 150,
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
