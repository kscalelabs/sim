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
    roll = "torso_joint"


class LeftArm(Node):
    shoulder_yaw = "left_shoulder_yaw_joint"
    shoulder_pitch = "left_shoulder_pitch_joint"
    shoulder_roll = "left_shoulder_roll_joint"
    elbow_pitch = "left_elbow_pitch_joint"


class RightArm(Node):
    shoulder_yaw = "right_shoulder_yaw_joint"
    shoulder_pitch = "right_shoulder_pitch_joint"
    shoulder_roll = "right_shoulder_roll_joint"
    elbow_pitch = "right_elbow_pitch_joint"


class LeftLeg(Node):
    hip_roll = "left_hip_roll_joint"
    hip_yaw = "left_hip_yaw_joint"
    hip_pitch = "left_hip_pitch_joint"
    knee_pitch = "left_knee_joint"
    ankle_pitch = "left_ankle_pitch_joint"
    ankle_roll = "left_ankle_roll_joint"


class RightLeg(Node):
    hip_roll = "right_hip_roll_joint"
    hip_yaw = "right_hip_yaw_joint"
    hip_pitch = "right_hip_pitch_joint"
    knee_pitch = "right_knee_joint"
    ankle_pitch = "right_ankle_pitch_joint"
    ankle_roll = "right_ankle_roll_joint"


class Legs(Node):
    left = LeftLeg()
    right = RightLeg()


class Robot(Node):
    height = 0.9

    torso = Torso()
    left_arm = LeftArm()
    right_arm = RightArm()
    legs = Legs()

    @classmethod
    def default_standing(cls) -> Dict[str, float]:
        return {  # = target angles [rad] when action = 0.0
            # left leg
            Robot.legs.left.hip_yaw: 0,
            Robot.legs.left.hip_roll: 0,
            Robot.legs.left.hip_pitch: -0.6,
            Robot.legs.left.knee_pitch: 1.2,
            Robot.legs.left.ankle_pitch: -0.6,
            Robot.legs.left.ankle_roll: 0.0,
            # right leg
            Robot.legs.right.hip_yaw: 0,
            Robot.legs.right.hip_roll: 0,
            Robot.legs.right.hip_pitch: -0.6,
            Robot.legs.right.knee_pitch: 1.2,
            Robot.legs.right.ankle_pitch: -0.6,
            Robot.legs.right.ankle_roll: 0.0,
            # torso
            Robot.torso.roll: 0,
            # left arm
            Robot.left_arm.shoulder_pitch: 0.4,
            Robot.left_arm.shoulder_roll: 0,
            Robot.left_arm.shoulder_yaw: 0,
            Robot.left_arm.elbow_pitch: 0.3,
            # right arm
            Robot.right_arm.shoulder_pitch: 0.4,
            Robot.right_arm.shoulder_roll: 0,
            Robot.right_arm.shoulder_yaw: 0,
            Robot.right_arm.elbow_pitch: 0.3,
        }

    # p_gains
    @classmethod
    def stiffness(cls) -> Dict[str, float]:
        return {
            "hip_pitch": 200,
            "hip_yaw": 200,
            "hip_roll": 200,
            "knee_joint": 300,
            "ankle_pitch": 40,
            "ankle_roll": 40,
            "shoulder_pitch": 80,
            "shoulder_yaw": 40,
            "shoulder_roll": 80,
            "elbow_pitch": 60,
            "torso_joint": 600,
        }

    # d_gains
    @classmethod
    def damping(cls) -> Dict[str, float]:
        return {
            "hip_pitch": 5,
            "hip_yaw": 5,
            "hip_roll": 5,
            "knee_joint": 7.5,
            "ankle_pitch": 1,
            "ankle_roll": 0.3,
            "shoulder_pitch": 2,
            "shoulder_yaw": 1,
            "shoulder_roll": 2,
            "elbow_pitch": 1,
            "torso_joint": 15,
        }


def print_joints() -> None:
    joints = Robot.all_joints()
    assert len(joints) == len(set(joints)), "Duplicate joint names found!"
    print(Robot())


if __name__ == "__main__":
    # python -m sim.Robot.joints
    print_joints()
