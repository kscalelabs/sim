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


class Hip(Node):
    class Pitch(Node):
        Left = "right_hip_pitch"
        Right = "left_hip_pitch"

    Lift_Joint_Limit_2 = "Hip_Lift_Joint_Limit_2"


class RightLeg(Node):
    hip_pitch = "right_hip_pitch"
    hip_lift = "right_hip_yaw"
    hip_roll = "right_hip_roll"
    knee_rotate = "right_knee_pitch"
    foot_rotate = "right_ankle_pitch"


class LeftLeg(Node):
    hip_pitch = "left_hip_pitch"
    hip_lift = "left_hip_yaw"
    hip_roll = "left_hip_roll"
    knee_rotate = "left_knee_pitch"
    foot_rotate = "left_ankle_pitch"


class LeftHand(Node):
    wrist_roll = "left hand roll"
    gripper = "left hand gripper"


class LeftArm(Node):
    shoulder_yaw = "left_shoulder_yaw"
    shoulder_pitch = "left_shoulder_pitch"
    shoulder_roll = "left_shoulder_roll"
    elbow_pitch = "left_elbow_pitch"
    # hand = LeftHand()  # Commented out


class RightHand(Node):
    wrist_roll = "right hand roll"
    gripper = "right hand gripper"


class RightArm(Node):
    shoulder_yaw = "right_shoulder_yaw"
    shoulder_pitch = "right_shoulder_pitch"
    shoulder_roll = "right_shoulder_roll"
    elbow_pitch = "right_elbow_pitch"
    # hand = RightHand()  # Commented out


class Legs(Node):
    left = LeftLeg()
    right = RightLeg()


class Robot(Node):
    height = 0.21
    rotation = [0.0, 0.0, 0, 1]

    # TODO
    # collision_links = [
    #     "lower_half_assembly_1_left_leg_1_foot_pad_1_simple",
    #     "lower_half_assembly_1_right_leg_1_foot_pad_1_simple",
    # ]

    legs = Legs()
    left_arm = LeftArm()
    right_arm = RightArm()

    @classmethod
    def default_standing(cls) -> Dict[str, float]:
        return {
            cls.legs.left.hip_pitch: 0,
            cls.legs.right.hip_pitch: 0,
            cls.legs.left.hip_lift: 0,
            cls.legs.right.hip_lift: 0,
            cls.legs.left.hip_roll: 0,
            cls.legs.right.hip_roll: 0,
            cls.legs.left.knee_rotate: 0,
            cls.legs.right.knee_rotate: 0,
            cls.legs.left.foot_rotate: 0,
            cls.legs.right.foot_rotate: 0,
            cls.left_arm.shoulder_yaw: 0,
            cls.right_arm.shoulder_yaw: 0,
            cls.left_arm.shoulder_pitch: 0,
            cls.right_arm.shoulder_pitch: 0,
            cls.left_arm.shoulder_roll: 0,
            cls.right_arm.shoulder_roll: 0,
            cls.left_arm.elbow_pitch: 0,
            cls.right_arm.elbow_pitch: 0,
            # cls.left_arm.hand.wrist_roll: 0,  # Commented out
            # cls.right_arm.hand.wrist_roll: 0,  # Commented out
            # cls.left_arm.hand.gripper: 0,  # Commented out
            # cls.right_arm.hand.gripper: 0,  # Commented out
        }

    @classmethod
    def default_limits(cls) -> Dict[str, Dict[str, float]]:
        return {
            # left leg
            cls.legs.left.hip_pitch: {
                "lower": -1.5707963,
                "upper": 1.5707963,
            },
            cls.legs.left.hip_lift: {
                "lower": -0.087266463,
                "upper": 1.5707963,
            },
            cls.legs.left.hip_roll: {
                "lower": -0.52359878,
                "upper": 0.52359878,
            },
            cls.legs.left.knee_rotate: {
                "lower": -1.5707963,
                "upper": 1.5707963,
            },
            cls.legs.left.foot_rotate: {
                "lower": -1.5707963,
                "upper": 1.5707963,
            },
            # right leg
            cls.legs.right.hip_pitch: {
                "lower": -1.5707963,
                "upper": 1.5707963,
            },
            cls.legs.right.hip_lift: {
                "lower": -1.5707963,
                "upper": 0.087266463,
            },
            cls.legs.right.hip_roll: {
                "lower": -0.52359878,
                "upper": 0.52359878,
            },
            cls.legs.right.knee_rotate: {
                "lower": -1.5707963,
                "upper": 1.5707963,
            },
            cls.legs.right.foot_rotate: {
                "lower": -1.5707963,
                "upper": 1.5707963,
            },
            # left arm
            cls.left_arm.shoulder_yaw: {
                "lower": -1.5707963,
                "upper": 1.5707963,
            },
            cls.left_arm.shoulder_pitch: {
                "lower": -1.5707963,
                "upper": 1.5707963,
            },
            cls.left_arm.shoulder_roll: {
                "lower": -1.5707963,
                "upper": 1.5707963,
            },
            cls.left_arm.elbow_pitch: {
                "lower": -1.5707963,
                "upper": 1.5707963,
            },
            # cls.left_arm.hand.wrist_roll: {  # Commented out
            #     "lower": -1.5707963,
            #     "upper": 1.5707963,
            # },
            # cls.left_arm.hand.gripper: {  # Commented out
            #     "lower": -1.5707963,
            #     "upper": 1.5707963,
            # },
            # right arm
            cls.right_arm.shoulder_yaw: {
                "lower": -1.5707963,
                "upper": 1.5707963,
            },
            cls.right_arm.shoulder_pitch: {
                "lower": -1.5707963,
                "upper": 1.5707963,
            },
            cls.right_arm.shoulder_roll: {
                "lower": -1.5707963,
                "upper": 1.5707963,
            },
            cls.right_arm.elbow_pitch: {
                "lower": -1.5707963,
                "upper": 1.5707963,
            },
            # cls.right_arm.hand.wrist_roll: {  # Commented out
            #     "lower": -1.5707963,
            #     "upper": 1.5707963,
            # },
            # cls.right_arm.hand.gripper: {  # Commented out
            #     "lower": -1.5707963,
            #     "upper": 1.5707963,
            # },
        }

    # p_gains
    @classmethod
    def stiffness(cls) -> Dict[str, float]:
        return {
            cls.legs.left.hip_pitch: 250,
            cls.legs.right.hip_pitch: 250,
            cls.legs.left.hip_lift: 150,
            cls.legs.right.hip_lift: 150,
            cls.legs.left.hip_roll: 250,
            cls.legs.right.hip_roll: 250,
            cls.legs.left.knee_rotate: 250,
            cls.legs.right.knee_rotate: 250,
            cls.legs.left.foot_rotate: 150,
            cls.legs.right.foot_rotate: 150,
            cls.left_arm.shoulder_yaw: 250,
            cls.right_arm.shoulder_yaw: 250,
            cls.left_arm.shoulder_pitch: 250,
            cls.right_arm.shoulder_pitch: 250,
            cls.left_arm.shoulder_roll: 250,
            cls.right_arm.shoulder_roll: 250,
            cls.left_arm.elbow_pitch: 250,
            cls.right_arm.elbow_pitch: 250,
            # cls.left_arm.hand.wrist_roll: 150,  # Commented out
            # cls.right_arm.hand.wrist_roll: 150,  # Commented out
            # cls.left_arm.hand.gripper: 150,  # Commented out
            # cls.right_arm.hand.gripper: 150,  # Commented out
        }

    # d_gains
    @classmethod
    def damping(cls) -> Dict[str, float]:
        return {
            cls.legs.left.hip_pitch: 10,
            cls.legs.right.hip_pitch: 10,
            cls.legs.left.hip_lift: 10,
            cls.legs.right.hip_lift: 10,
            cls.legs.left.hip_roll: 10,
            cls.legs.right.hip_roll: 10,
            cls.legs.left.knee_rotate: 10,
            cls.legs.right.knee_rotate: 10,
            cls.legs.left.foot_rotate: 10,
            cls.legs.right.foot_rotate: 10,
            cls.left_arm.shoulder_yaw: 10,
            cls.right_arm.shoulder_yaw: 10,
            cls.left_arm.shoulder_pitch: 10,
            cls.right_arm.shoulder_pitch: 10,
            cls.left_arm.shoulder_roll: 10,
            cls.right_arm.shoulder_roll: 10,
            cls.left_arm.elbow_pitch: 10,
            cls.right_arm.elbow_pitch: 10,
            # cls.left_arm.hand.wrist_roll: 10,  # Commented out
            # cls.right_arm.hand.wrist_roll: 10,  # Commented out
            # cls.left_arm.hand.gripper: 10,  # Commented out
            # cls.right_arm.hand.gripper: 10,  # Commented out
        }

    # pos_limits
    @classmethod
    def effort(cls) -> Dict[str, float]:
        return {
            "hip pitch": 80,
            "hip lift": 80,
            "hip roll": 80,
            "knee rotate": 80,
            "foot rotate": 80,
            "shoulder yaw": 80,
            "shoulder pitch": 80,
            "shoulder roll": 80,
            "elbow pitch": 80,
            # "wrist roll": 80,
            # "gripper": 80,
        }

    # vel_limits
    @classmethod
    def velocity(cls) -> Dict[str, float]:
        return {
            "hip pitch": 5,
            "hip lift": 5,
            "hip roll": 5,
            "knee rotate": 5,
            "foot rotate": 5,
            "shoulder yaw": 5,
            "shoulder pitch": 5,
            "shoulder roll": 5,
            "elbow pitch": 5,
            # "wrist roll": 5,
            # "gripper": 5,
        }

    @classmethod
    def friction(cls) -> Dict[str, float]:
        return {
            "hip pitch": 0.0,
            "hip lift": 0.0,
            "hip roll": 0.0,
            "knee rotate": 0.0,
            "foot rotate": 0.0,
            "shoulder yaw": 0.0,
            "shoulder pitch": 0.0,
            "shoulder roll": 0.0,
            "elbow pitch": 0.0,
            # "wrist roll": 0.0,
            # "gripper": 0.0,
        }


def print_joints() -> None:
    joints = Robot.all_joints()
    assert len(joints) == len(set(joints)), "Duplicate joint names found!"
    print(Robot())


if __name__ == "__main__":
    print_joints()
