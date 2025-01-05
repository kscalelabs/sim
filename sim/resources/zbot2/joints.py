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
    hip_roll = "right_hip_roll"
    hip_yaw = "right_hip_yaw"
    hip_pitch = "right_hip_pitch"
    knee_pitch = "right_knee_pitch"
    ankle_pitch = "right_ankle_pitch"


class LeftLeg(Node):
    hip_roll = "left_hip_roll"
    hip_yaw = "left_hip_yaw"
    hip_pitch = "left_hip_pitch"
    knee_pitch = "left_knee_pitch"
    ankle_pitch = "left_ankle_pitch"


class Legs(Node):
    right = RightLeg()
    left = LeftLeg()


class Robot(Node):
    legs = Legs()

    height = 0.40
    standing_height = 0.407
    rotation = [0, 0, 0, 1.0]

    @classmethod
    def default_walking(cls) -> Dict[str, float]:
        """Example angles for a nominal 'standing' pose. Adjust as needed."""
        return {
            # Right Leg
            cls.legs.right.hip_roll: 0.0,
            cls.legs.right.hip_yaw: 0.0,
            cls.legs.right.hip_pitch: -0.296,
            cls.legs.right.knee_pitch: 0.579,
            cls.legs.right.ankle_pitch: 0.283,
            # Left Leg
            cls.legs.left.hip_roll: 0.0,
            cls.legs.left.hip_yaw: 0.0,
            cls.legs.left.hip_pitch: 0.296,
            cls.legs.left.knee_pitch: 2.2,
            cls.legs.left.ankle_pitch: 0.927,
        }

    @classmethod
    def default_standing(cls) -> Dict[str, float]:
        """Example angles for a nominal 'standing' pose. Adjust as needed."""
        return {
            # Right Leg
            cls.legs.right.hip_roll: 0.0,
            cls.legs.right.hip_yaw: 0.0,
            cls.legs.right.hip_pitch: 0.0,
            cls.legs.right.knee_pitch: 0.0,
            cls.legs.right.ankle_pitch: 0.0,
            # Left Leg
            cls.legs.left.hip_roll: 0.0,
            cls.legs.left.hip_yaw: 0.0,
            cls.legs.left.hip_pitch: 0.0,
            cls.legs.left.knee_pitch: 2.76,
            cls.legs.left.ankle_pitch: 1.2,
        }

    @classmethod
    def default_limits(cls) -> Dict[str, Dict[str, float]]:
        """Minimal example of per-joint limits.

        You can refine these to match your MJCF's 'range' tags or real specs.
        """
        return {
            # Right side
            # cls.arms.right.shoulder_yaw: {"lower": -1.22173, "upper": 0.349066},
            # cls.arms.right.shoulder_pitch: {"lower": -3.14159, "upper": 3.14159},
            # cls.arms.right.elbow_yaw: {"lower": -1.5708, "upper": 2.0944},
            # cls.arms.right.gripper: {"lower": -0.349066, "upper": 0.872665},
            cls.legs.right.hip_roll: {"lower": -1.0472, "upper": 1.0472},
            cls.legs.right.hip_yaw: {"lower": -0.174533, "upper": 1.91986},
            cls.legs.right.hip_pitch: {"lower": -1.74533, "upper": 1.8326},
            cls.legs.right.knee_pitch: {"lower": -0.174533, "upper": 2.96706},
            cls.legs.right.ankle_pitch: {"lower": -1.5708, "upper": 1.5708},
            # Left side
            # cls.arms.left.shoulder_yaw: {"lower": -0.349066, "upper": 1.22173},
            # cls.arms.left.shoulder_pitch: {"lower": -3.14159, "upper": 3.14159},
            # cls.arms.left.elbow_yaw: {"lower": -2.0944, "upper": 1.5708},
            # cls.arms.left.gripper: {"lower": -0.872665, "upper": 0.349066},
            cls.legs.left.hip_roll: {"lower": -1.0472, "upper": 1.0472},
            cls.legs.left.hip_yaw: {"lower": -1.91986, "upper": 0.174533},
            cls.legs.left.hip_pitch: {"lower": -1.8326, "upper": 1.74533},
            cls.legs.left.knee_pitch: {"lower": -0.174533, "upper": 2.96706},
            cls.legs.left.ankle_pitch: {"lower": -1.5708, "upper": 1.5708},
        }

    # p_gains
    @classmethod
    def stiffness(cls) -> Dict[str, float]:
        return {
            "hip_pitch": 5,
            "hip_yaw": 5,
            "hip_roll": 5,
            "knee_pitch": 5,
            "ankle_pitch": 5,
        }

    # d_gains
    @classmethod
    def damping(cls) -> Dict[str, float]:
        return {
            "hip_pitch": 1.0,
            "hip_yaw": 1.0,
            "hip_roll": 1.0,
            "knee_pitch": 1.0,
            "ankle_pitch": 1.0,
        }

    @classmethod
    def effort(cls) -> Dict[str, float]:
        return {
            "hip_pitch": 0.3,
            "hip_yaw": 0.3,
            "hip_roll": 0.3,
            "knee_pitch": 0.3,
            "ankle_pitch": 0.3,
        }

    # vel_limits
    @classmethod
    def velocity(cls) -> Dict[str, float]:
        return {
            "hip_pitch": 20,
            "hip_yaw": 20,
            "hip_roll": 20,
            "knee_pitch": 20,
            "ankle_pitch": 20,
        }

    @classmethod
    def friction(cls) -> Dict[str, float]:
        """Example friction dictionary for certain joints."""
        # Usually you'd have more specific friction values or a model.
        return {
            cls.legs.left.ankle_pitch: 0.01,
            cls.legs.right.ankle_pitch: 0.01,
            # etc...
        }


def print_joints() -> None:
    # Gather all joints and check for duplicates
    joints_list = Robot.all_joints()
    assert len(joints_list) == len(set(joints_list)), "Duplicate joint names found!"

    # Print out the structure for debugging
    print(Robot())
    print("\nAll Joints:", joints_list)


if __name__ == "__main__":
    print_joints()
