"""Provides a Pythonic interface for referencing joint names from the given MuJoCo XML.

Organizes them by sub-assembly (arms, legs) and defines convenient methods for
defaults, limits, etc.
"""

import textwrap
from abc import ABC
from typing import Dict, List, Tuple, Union


class Node(ABC):
    @classmethod
    def children(cls) -> List["Union[Node, str]"]:
        # Returns Node or string attributes (for recursion)
        return [
            attr
            for attr in (getattr(cls, attr_name) for attr_name in dir(cls) if not attr_name.startswith("__"))
            if isinstance(attr, (Node, str))
        ]

    @classmethod
    def joints(cls) -> List[str]:
        # Returns only the attributes that are strings (i.e., joint names).
        return [
            attr
            for attr in (getattr(cls, attr_name) for attr_name in dir(cls) if not attr_name.startswith("__"))
            if isinstance(attr, str)
        ]

    @classmethod
    def joints_motors(cls) -> List[Tuple[str, str]]:
        # Returns pairs of (attribute_name, joint_string)
        joint_names: List[Tuple[str, str]] = []
        for attr_name in dir(cls):
            if not attr_name.startswith("__"):
                attr = getattr(cls, attr_name)
                if isinstance(attr, str):
                    joint_names.append((attr_name, attr))
        return joint_names

    @classmethod
    def all_joints(cls) -> List[str]:
        # Recursively collect all string joint names
        leaves = cls.joints()
        for child in cls.children():
            if isinstance(child, Node):
                leaves.extend(child.all_joints())
        return leaves

    def __str__(self) -> str:
        # Pretty-print the hierarchy
        parts = [str(child) for child in self.children() if isinstance(child, Node)]
        parts_str = textwrap.indent("\n" + "\n".join(parts), "  ") if parts else ""
        return f"[{self.__class__.__name__}]{parts_str}"


# ----- Define the sub-assemblies ---------------------------------------------------
class RightArm(Node):
    shoulder_yaw = "right_shoulder_yaw"
    shoulder_pitch = "right_shoulder_pitch"
    elbow_yaw = "right_elbow_yaw"
    gripper = "right_gripper"


class LeftArm(Node):
    shoulder_yaw = "left_shoulder_yaw"
    shoulder_pitch = "left_shoulder_pitch"
    elbow_yaw = "left_elbow_yaw"
    gripper = "left_gripper"


class Arms(Node):
    right = RightArm()
    left = LeftArm()


class RightLeg(Node):
    hip_roll = "R_Hip_Roll"
    hip_yaw = "R_Hip_Yaw"
    hip_pitch = "R_Hip_Pitch"
    knee_pitch = "R_Knee_Pitch"
    ankle_pitch = "R_Ankle_Pitch"


class LeftLeg(Node):
    hip_roll = "L_Hip_Roll"
    hip_yaw = "L_Hip_Yaw"
    hip_pitch = "L_Hip_Pitch"
    knee_pitch = "L_Knee_Pitch"
    ankle_pitch = "L_Ankle_Pitch"


class Legs(Node):
    right = RightLeg()
    left = LeftLeg()


class Robot(Node):
    legs = Legs()

    height = 0.38
    standing_height = 0.407
    rotation = [0, 0, 0, 1.0]

    @classmethod
    def default_walking(cls) -> Dict[str, float]:
        """Example angles for a nominal 'standing' pose. Adjust as needed."""
        return {
            # Left Leg
            cls.legs.left.hip_roll: 0.0,
            cls.legs.left.hip_yaw: 0.0,
            cls.legs.left.hip_pitch: -0.689,
            cls.legs.left.knee_pitch: 1.44,
            cls.legs.left.ankle_pitch: 0.747,
            # Right Leg
            cls.legs.right.hip_roll: 0.0,
            cls.legs.right.hip_yaw: 0.0,
            cls.legs.right.hip_pitch: 0.689,
            cls.legs.right.knee_pitch: -1.44,
            cls.legs.right.ankle_pitch: -0.747,
        }

    @classmethod
    def default_standing(cls) -> Dict[str, float]:
        """Example angles for a nominal 'standing' pose. Adjust as needed."""
        return {
            # Left Leg
            cls.legs.left.hip_roll: 0.0,
            cls.legs.left.hip_yaw: 0.0,
            cls.legs.left.hip_pitch: 0.0,
            cls.legs.left.knee_pitch: 0.0,
            cls.legs.left.ankle_pitch: 0.0,
            # Right Leg
            cls.legs.right.hip_roll: 0.0,
            cls.legs.right.hip_yaw: 0.0,
            cls.legs.right.hip_pitch: 0.0,
            cls.legs.right.knee_pitch: 0.0,
            cls.legs.right.ankle_pitch: 0.0,
        }

    # CONTRACT - this should be ordered according to how the policy is trained.
    # E.g. the first entry should be the name of the first joint in the policy.
    @classmethod
    def joint_names(cls) -> List[str]:
        return list(cls.default_standing().keys())

    @classmethod
    def default_limits(cls) -> Dict[str, Dict[str, float]]:
        """Minimal example of per-joint limits.

        You can refine these to match your MJCF's 'range' tags or real specs.
        """
        return {
            # Left side
            cls.legs.left.hip_roll: {"lower": -0.7, "upper": 0.7},
            cls.legs.left.hip_yaw: {"lower": -1.54, "upper": 0.157},
            cls.legs.left.hip_pitch: {"lower": -1.54, "upper": 1.0},
            cls.legs.left.knee_pitch: {"lower": 0, "upper": 1.57},
            cls.legs.left.ankle_pitch: {"lower": -0.9, "upper": 0.9},
            # Right side
            cls.legs.right.hip_roll: {"lower": -0.7, "upper": 0.7},
            cls.legs.right.hip_yaw: {"lower": -1.54, "upper": 0.157},
            cls.legs.right.hip_pitch: {"lower": -1.0, "upper": 1.54},
            cls.legs.right.knee_pitch: {"lower": -1.57, "upper": 0},
            cls.legs.right.ankle_pitch: {"lower": -0.9, "upper": 0.9},
        }

    # p_gains
    @classmethod
    def stiffness(cls) -> Dict[str, float]:
        return {
            "Hip_Pitch": 18.18,
            "Hip_Yaw": 18.18,
            "Hip_Roll": 18.18,
            "Knee_Pitch": 18.18,
            "Ankle_Pitch": 18.18,
        }

    # d_gains
    @classmethod
    def damping(cls) -> Dict[str, float]:
        return {
            "Hip_Pitch": 1.46,
            "Hip_Yaw": 1.46,
            "Hip_Roll": 1.46,
            "Knee_Pitch": 1.46,
            "Ankle_Pitch": 1.46,
        }

    @classmethod
    def effort(cls) -> Dict[str, float]:
        return {
            "Hip_Pitch": 1.91,
            "Hip_Yaw": 1.91,
            "Hip_Roll": 1.91,
            "Knee_Pitch": 1.91,
            "Ankle_Pitch": 1.91,
        }

    # vel_limits
    @classmethod
    def velocity(cls) -> Dict[str, float]:
        return {
            "Hip_Pitch": 2,
            "Hip_Yaw": 2,
            "Hip_Roll": 2,
            "Knee_Pitch": 2,
            "Ankle_Pitch": 2,
        }

    @classmethod
    def friction(cls) -> Dict[str, float]:
        """Example friction dictionary for certain joints."""
        # Usually you'd have more specific friction values or a model.
        return {
            "Hip_Pitch": 0.0,
            "Hip_Yaw": 0.0,
            "Hip_Roll": 0.0,
            "Knee_Pitch": 0.0,
            "Ankle_Pitch": 0.01,
        }

    @classmethod
    def effort_mapping(cls) -> Dict[str, float]:
        mapping = {}
        effort = cls.effort()
        for side in ["left", "right"]:
            for joint, value in effort.items():
                mapping[f"{side}_{joint}"] = value
        return mapping

    @classmethod
    def stiffness_mapping(cls) -> Dict[str, float]:
        mapping = {}
        stiffness = cls.stiffness()
        for side in ["left", "right"]:
            for joint, value in stiffness.items():
                mapping[f"{side}_{joint}"] = value
        return mapping

    @classmethod
    def damping_mapping(cls) -> Dict[str, float]:
        mapping = {}
        damping = cls.damping()
        for side in ["left", "right"]:
            for joint, value in damping.items():
                mapping[f"{side}_{joint}"] = value
        return mapping


def print_joints() -> None:
    # Gather all joints and check for duplicates
    joints_list = Robot.all_joints()
    assert len(joints_list) == len(set(joints_list)), "Duplicate joint names found!"

    # Print out the structure for debugging
    print(Robot())
    print("\nAll Joints:", joints_list)


if __name__ == "__main__":
    print_joints()
