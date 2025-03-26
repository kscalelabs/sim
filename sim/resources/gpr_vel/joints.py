"""Defines a more Pythonic interface for specifying the joint names."""

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
    ankle_pitch = "L_ankle"


class RightLeg(Node):
    hip_pitch = "R_hip_y"
    hip_yaw = "R_hip_x"
    hip_roll = "R_hip_z"
    knee_pitch = "R_knee"
    ankle_pitch = "R_ankle"


class Legs(Node):
    left = LeftLeg()
    right = RightLeg()


class Robot(Node):
    legs = Legs()

    height = 1.05
    standing_height = 1.05 + 0.025
    rotation = [0, 0, 0, 1]

    @classmethod
    def isaac_to_real_signs(cls) -> Dict[str, int]:
        return {
            Robot.legs.left.hip_pitch: 1,
            Robot.legs.left.hip_yaw: 1,
            Robot.legs.left.hip_roll: 1,
            Robot.legs.left.knee_pitch: 1,
            Robot.legs.left.ankle_pitch: 1,
            Robot.legs.right.hip_pitch: -1,
            Robot.legs.right.hip_yaw: -1,
            Robot.legs.right.hip_roll: 1,
            Robot.legs.right.knee_pitch: -1,
            Robot.legs.right.ankle_pitch: 1,
        }

    @classmethod
    def isaac_to_mujoco_signs(cls) -> Dict[str, int]:
        return {
            Robot.legs.left.hip_pitch: 1,
            Robot.legs.left.hip_yaw: 1,
            Robot.legs.left.hip_roll: 1,
            Robot.legs.left.knee_pitch: 1,
            Robot.legs.left.ankle_pitch: 1,
            Robot.legs.right.hip_pitch: 1,
            Robot.legs.right.hip_yaw: 1,
            Robot.legs.right.hip_roll: 1,
            Robot.legs.right.knee_pitch: 1,
            Robot.legs.right.ankle_pitch: 1,
        }

    @classmethod
    def default_positions(cls) -> Dict[str, float]:
        return {
            Robot.legs.left.hip_pitch: 0.0,
            Robot.legs.left.hip_yaw: 0.0,
            Robot.legs.left.hip_roll: 0.0,
            Robot.legs.left.knee_pitch: 0.0,
            Robot.legs.left.ankle_pitch: 0.0,
            Robot.legs.right.hip_pitch: 0.0,
            Robot.legs.right.hip_yaw: 0.0,
            Robot.legs.right.hip_roll: 0.0,
            Robot.legs.right.knee_pitch: 0.0,
            Robot.legs.right.ankle_pitch: 0.0,
        }

    # CONTRACT - this should be ordered according to how the policy is trained.
    # E.g. the first entry should be the angle of the first joint in the policy.
    @classmethod
    def default_standing(cls) -> Dict[str, float]:
        return {
            Robot.legs.left.hip_pitch: 0.23,
            Robot.legs.left.hip_yaw: 0.0,
            Robot.legs.left.hip_roll: 0.0,
            Robot.legs.left.knee_pitch: -0.441,
            Robot.legs.left.ankle_pitch: -0.195,
            Robot.legs.right.hip_pitch: -0.23,
            Robot.legs.right.hip_yaw: 0.0,
            Robot.legs.right.hip_roll: 0.0,
            Robot.legs.right.knee_pitch: -0.441,
            Robot.legs.right.ankle_pitch: -0.195,
        }

    # CONTRACT - this should be ordered according to how the policy is trained.
    # E.g. the first entry should be the name of the first joint in the policy.
    @classmethod
    def joint_names(cls) -> List[str]:
        return list(cls.default_standing().keys())

    @classmethod
    def default_limits(cls) -> Dict[str, Dict[str, float]]:
        return {
            Robot.legs.left.knee_pitch: {"lower": -1.57, "upper": 0},
            Robot.legs.right.knee_pitch: {"lower": -1.57, "upper": 0},
        }

    # p_gains
    @classmethod
    def stiffness(cls) -> Dict[str, float]:
        return {
            "hip_y": 85,
            "hip_x": 40,
            "hip_z": 40,
            "knee": 85,
            "ankle": 30,
        }

    @classmethod
    def stiffness_mapping(cls) -> Dict[str, float]:
        mapping = {}
        stiffness = cls.stiffness()
        for side in ["L", "R"]:
            for joint, value in stiffness.items():
                mapping[f"{side}_{joint}"] = value
        return mapping

    # d_gains
    @classmethod
    def damping(cls) -> Dict[str, float]:
        return {
            "hip_y": 5,
            "hip_x": 4,
            "hip_z": 4,
            "knee": 5,
            "ankle": 1,
        }

    @classmethod
    def damping_mapping(cls) -> Dict[str, float]:
        mapping = {}
        damping = cls.damping()
        for side in ["L", "R"]:
            for joint, value in damping.items():
                mapping[f"{side}_{joint}"] = value
        return mapping

    # effort_limits
    @classmethod
    def effort(cls) -> Dict[str, float]:
        return {
            "hip_y": 60,
            "hip_x": 40,
            "hip_z": 40,
            "knee": 60,
            "ankle": 17,
        }

    @classmethod
    def effort_mapping(cls) -> Dict[str, float]:
        mapping = {}
        effort = cls.effort()
        for side in ["L", "R"]:
            for joint, value in effort.items():
                mapping[f"{side}_{joint}"] = value
        return mapping

    # vel_limits
    @classmethod
    def velocity(cls) -> Dict[str, float]:
        return {"hip": 1000, "knee": 1000, "ankle": 1000}

    @classmethod
    def friction(cls) -> Dict[str, float]:
        return {
            "hip": 0,
            "knee": 0,
            "ankle": 0.1,
        }


def print_joints() -> None:
    joints = Robot.all_joints()
    assert len(joints) == len(set(joints)), "Duplicate joint names found!"
    print(Robot())
    print(len(joints))


if __name__ == "__main__":
    # python -m sim.Robot.joints
    print_joints()
