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


class LeftLeg(Node):
    hip_pitch = "L_hip_y"
    hip_yaw = "L_hip_x"
    hip_roll = "L_hip_z"
    knee_pitch = "L_knee"
    ankle_pitch = "L_ankle_y"


class RightLeg(Node):
    hip_pitch = "R_hip_y"
    hip_yaw = "R_hip_x"
    hip_roll = "R_hip_z"
    knee_pitch = "R_knee"
    ankle_pitch = "R_ankle_y"


class LeftArm(Node):
    shoulder_pitch = "L_shoulder_y"
    shoulder_roll = "L_shoulder_z"
    shoulder_yaw = "L_shoulder_x"
    elbow_pitch = "L_elbow_x"


class RightArm(Node):
    shoulder_pitch = "R_shoulder_y"
    shoulder_roll = "R_shoulder_z"
    shoulder_yaw = "R_shoulder_x"
    elbow_pitch = "R_elbow_x"


class Legs(Node):
    left = LeftLeg()
    right = RightLeg()


class Arms(Node):
    left = LeftArm()
    right = RightArm()


class Robot(Node):
    legs = Legs()
    # arms = Arms()

    collision_links: List[str] = [] # None because StompyPro feet are boxes

    height = 0.63
    rotation = [0.0, 0.0, 0, 1]

    @classmethod
    def default_positions(cls) -> Dict[str, float]:
        return {}

    @classmethod
    def default_standing(cls) -> Dict[str, float]:
        return {
            Robot.legs.left.hip_pitch: -0.157,
            Robot.legs.left.hip_yaw: 0.0394,
            Robot.legs.left.hip_roll: 0.0628,
            Robot.legs.left.knee_pitch: 0.441,
            Robot.legs.left.ankle_pitch: -0.258,
            Robot.legs.right.hip_pitch: -0.22,
            Robot.legs.right.hip_yaw: 0.026,
            Robot.legs.right.hip_roll: 0.0314,
            Robot.legs.right.knee_pitch: 0.441,
            Robot.legs.right.ankle_pitch: -0.223,
        }

    # @classmethod
    # def default_limits(cls) -> Dict[str, Dict[str, float]]:
    #     return {
    #         Robot.legs.left.hip_pitch: {
    #             "lower": 0.5,
    #             "upper": 2.69,
    #         },
    #         Robot.legs.left.hip_yaw: {
    #             "lower": 0.5,
    #             "upper": 1.19,
    #         },
    #         Robot.legs.left.hip_roll: {
    #             "lower": -0.5,
    #             "upper": 0.5,
    #         },
    #         Robot.legs.left.knee_pitch: {
    #             "lower": -2.14,
    #             "upper": -1.0,
    #         },
    #         Robot.legs.left.ankle_pitch: {
    #             "lower": -0.8,
    #             "upper": 0.6,
    #         },
    #         Robot.legs.right.hip_pitch: {
    #             "lower": -1,
    #             "upper": 1,
    #         },
    #         Robot.legs.right.hip_yaw: {
    #             "lower": -2.6,
    #             "upper": -1.5,
    #         },
    #         Robot.legs.right.hip_roll: {
    #             "lower": -2.39,
    #             "upper": -1,
    #         },
    #         Robot.legs.right.knee_pitch: {
    #             "lower": 2.09,
    #             "upper": 3.2,
    #         },
    #         Robot.legs.right.ankle_pitch: {
    #             "lower": 0,
    #             "upper": 1.5,
    #         },
    #     }

    @classmethod
    def default_limits(cls) -> Dict[str, Dict[str, float]]:
        return {}

    # p_gains
    @classmethod
    def stiffness(cls) -> Dict[str, float]:
        return {
            "hip_y": 120,
            "hip_x": 60,
            "hip_z": 60,
            "knee": 120,
            "ankle_y": 17,
        }

    # d_gains
    @classmethod
    def damping(cls) -> Dict[str, float]:
        return {
            "hip_y": 10,
            "hip_x": 10,
            "hip_z": 10,
            "knee": 10,
            "ankle_y": 5,
        }

    # pos_limits
    @classmethod
    def effort(cls) -> Dict[str, float]:
        return {}

    # vel_limits
    @classmethod
    def velocity(cls) -> Dict[str, float]:
        return {
            "hip": 5,
            "knee": 5,
            "ankle": 5,
            "shoulder_y": 5,
            "shoulder_z": 5,
            "shoulder_x": 5,
            "elbow_x": 5,
        }

    @classmethod
    def friction(cls) -> Dict[str, float]:
        return {
            "hip": 0,
            "knee": 0,
            "ankle": 0,
        }


def print_joints() -> None:
    joints = Robot.all_joints()
    assert len(joints) == len(set(joints)), "Duplicate joint names found!"
    print(Robot())
    print(len(joints))


if __name__ == "__main__":
    # python -m sim.Robot.joints
    print_joints()
