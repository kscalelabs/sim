"""Defines a more Pythonic interface for specifying the joint names.

The best way to re-generate this snippet for a new robot is to use the
`sim/scripts/print_joints.py` script. This script will print out a hierarchical
tree of the various joint names in the robot.
"""
"""
The OUTPUT:

Left
  Elbow_Pitch: Left_Elbow_Pitch
  Hip_Pitch: Left_Hip_Pitch
  Knee_Pitch: Left_Knee_Pitch
  Shoulder_Pitch: Left_Shoulder_Pitch
Right
  Elbow_Pitch: Right_Elbow_Pitch
  Hip_Pitch: Right_Hip_Pitch
  Knee_Pitch: Right_Knee_Pitch
  Shoulder_Pitch: Right_Shoulder_Pitch

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
#     wrist_roll = "left hand roll"
#     gripper = "left hand gripper"


class LeftArm(Node):
    shoulder_pitch = "Left_Shoulder_Pitch"
    elbow_pitch = "Left_Elbow_Pitch"


# class RightHand(Node):
#     wrist_roll = "right hand roll"
#     gripper = "right hand gripper"


class RightArm(Node):
    shoulder_pitch = "Right_Shoulder_Pitch"
    elbow_pitch = "Right_Elbow_Pitch"


class LeftLeg(Node):
    hip_pitch = "Left_Hip_Pitch"
    knee_pitch = "Left_Knee_Pitch"


class RightLeg(Node):
    hip_pitch = "Right_Hip_Pitch"
    knee_pitch = "Right_Knee_Pitch"


class Legs(Node):
    left = LeftLeg()
    right = RightLeg()

class Arms(Node):
    left = LeftArm()
    right = RightArm()


class Robot(Node):
    # STEP TWO
    height = 0.256
    rotation = [0.5000, -0.4996, -0.5000, 0.5004] #Which orientation the robot itself is
    collision_links = [
        "lower_half_assembly_1_left_leg_1_foot_pad_1_simple",
        "lower_half_assembly_1_right_leg_1_foot_pad_1_simple",
    ] #not important/ no need to update?

    arms = Arms()
    legs = Legs()

    @classmethod
    def default_standing(cls) -> Dict[str, float]:
        return {
            # arms
            Robot.arms.left.shoulder_pitch: 0.57,
            Robot.arms.left.elbow_pitch: -1.40,
        
            Robot.arms.right.shoulder_pitch: 0.57,
            Robot.arms.right.elbow_pitch: 1.40,

            # legs
            Robot.legs.left.hip_pitch: 0.90,
            Robot.legs.left.knee_pitch: -1.40,
        
            Robot.legs.right.hip_pitch: 0.90,
            Robot.legs.right.knee_pitch: -1.40,
        }

    @classmethod
    def default_limits2(cls) -> Dict[str, Dict[str, float]]:
        return {
            # left arm
            Robot.arms.left.Left_Shoulder_Pitch: {
                "lower": 0,
                "upper": 1.22,
            },
            Robot.arms.left.Left_Elbow_Pitch: {
                "lower": -1.40,
                "upper": 1.40,
            },
            # right arm
            Robot.arms.right.Right_Shoulder_Pitch:  {
                "lower": 0,
                "upper": 1.22,
            },
            Robot.arms.right.Right_Elbow_Pitch:  {
                "lower": -1.40,
                "upper": 1.40,
            },
            # left leg
            Robot.legs.left.Left_Hip_Pitch:  {
                "lower": 0, 
                "upper": 1.57,
            },
            Robot.legs.left.Left_Knee_Pitch: {
                "lower": -1.40,
                "upper": 1.40,
            },
            # right leg
            Robot.legs.right.Right_Hip_Pitch: {
                "lower": 0,
                "upper": 1.57,
            },
            Robot.legs.right.Right_Knee_Pitch: {
                "lower": -1.40,
                "upper": 1.40,
            }
        }

    @classmethod
    def default_limits(cls) -> Dict[str, Dict[str, float]]:
         return {
            # left arm
            Robot.arms.left.Left_Shoulder_Pitch: {
                "lower": 0,
                "upper": 1.22,
            },
            Robot.arms.left.Left_Elbow_Pitch: {
                "lower": -1.40,
                "upper": 1.40,
            },
            # right arm
            Robot.arms.right.Right_Shoulder_Pitch:  {
                "lower": 0,
                "upper": 1.22,
            },
            Robot.arms.right.Right_Elbow_Pitch:  {
                "lower": -1.40,
                "upper": 1.40,
            },
            # left leg
            Robot.legs.left.Left_Hip_Pitch:  {
                "lower": 0,
                "upper": 1.22,
            },
            Robot.legs.left.Left_Knee_Pitch: {
                "lower": -1.40,
                "upper": 1.40,
            },
            # right leg
            Robot.legs.right.Right_Hip_Pitch: {
                "lower": 0,
                "upper": 1.22,
            },
            Robot.legs.right.Right_Knee_Pitch: {
                "lower": -1.40,
                "upper": 1.40,
            }
        }

    # p_gains
    @classmethod
    def stiffness(cls) -> Dict[str, float]:
        return {
            "hip pitch": 250,
            "hip yaw": 250,
            "hip roll": 150,
            "knee pitch": 250,
            "ankle pitch": 150,
            "shoulder pitch": 150,
            "shoulder yaw": 45,
            "shoulder roll": 45,
            "elbow pitch": 45,
            "hand roll": 45,
            "gripper": 45,
        }

    # d_gains
    @classmethod
    def damping(cls) -> Dict[str, float]:
        return {
            "hip pitch": 10,
            "hip yaw": 10,
            "hip roll": 10,
            "knee pitch": 10,
            "ankle pitch": 10,
            "shoulder pitch": 10,
            "shoulder yaw": 10,
            "shoulder roll": 5,
            "elbow pitch": 5,
            "hand roll": 5,
            "gripper": 5,
        }

    # pos_limits
    @classmethod
    def effort(cls) -> Dict[str, float]:
        return {
            "hip pitch": 120,
            "hip yaw": 120,
            "hip roll": 17,
            "knee pitch": 120,
            "ankle pitch": 17,
            "shoulder pitch": 120,
            "shoulder yaw": 17,
            "shoulder roll": 17,
            "elbow pitch": 17,
            "hand roll": 17,
            "gripper": 17,
        }

    # vel_limits
    @classmethod
    def velocity(cls) -> Dict[str, float]:
        return {
            "hip pitch": 40,
            "hip yaw": 40,
            "hip roll": 40,
            "knee pitch": 40,
            "ankle pitch": 40,
            "shoulder pitch": 40,
            "shoulder yaw": 40,
            "shoulder roll": 40,
            "elbow pitch": 40,
            "hand roll": 40,
            "gripper": 40,
        }

    @classmethod
    def friction(cls) -> Dict[str, float]:
        return {
            "hip pitch": 0.0,
            "hip yaw": 0.0,
            "hip roll": 0.0,
            "knee pitch": 0.0,
            "ankle pitch": 0.0,
            "hand roll": 0.0,
            "gripper": 0.0,
        }


def print_joints() -> None:
    joints = Robot.all_joints()
    assert len(joints) == len(set(joints)), "Duplicate joint names found!"
    print(Robot())


if __name__ == "__main__":
    # python -m sim.Robot.joints
    print_joints()
