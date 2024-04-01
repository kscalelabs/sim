"""Defines a more Pythonic interface for specifying the joint names.

The best way to re-generate this snippet for a new robot is to use the
`sim/scripts/print_joints.py` script. This script will print out a hierarchical
tree of the various joint names in the robot.
"""

import textwrap
from abc import ABC
from typing import List, Union


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


class Head(Node):
    left_right: str = "joint_head_1_x4_1_dof_x4"
    up_down: str = "joint_head_1_x4_2_dof_x4"


class Torso(Node):
    pitch: str = "joint_torso_1_x8_1_dof_x8"


class LeftHand(Node):
    hand_roll: str = "joint_left_arm_2_hand_1_x4_1_dof_x4"
    hand_grip: str = "joint_left_arm_2_hand_1_x4_2_dof_x4"
    slider_a: str = "joint_left_arm_2_hand_1_slider_1"
    slider_b: str = "joint_left_arm_2_hand_1_slider_2"


class LeftArm(Node):
    shoulder_yaw: str = "joint_left_arm_2_x8_1_dof_x8"
    shoulder_pitch: str = "joint_left_arm_2_x8_2_dof_x8"
    shoulder_roll: str = "joint_left_arm_2_x6_1_dof_x6"
    elbow_yaw: str = "joint_left_arm_2_x6_2_dof_x6"
    elbow_roll: str = "joint_left_arm_2_x4_1_dof_x4"
    hand: Node = LeftHand()


class RightHand(Node):
    hand_roll: str = "joint_right_arm_1_hand_1_x4_1_dof_x4"
    hand_grip: str = "joint_right_arm_1_hand_1_x4_2_dof_x4"
    slider_a: str = "joint_right_arm_1_hand_1_slider_1"
    slider_b: str = "joint_right_arm_1_hand_1_slider_2"


class RightArm(Node):
    shoulder_yaw: str = "joint_right_arm_1_x8_1_dof_x8"
    shoulder_pitch: str = "joint_right_arm_1_x8_2_dof_x8"
    shoulder_roll: str = "joint_right_arm_1_x6_1_dof_x6"
    elbow_yaw: str = "joint_right_arm_1_x6_2_dof_x6"
    elbow_roll: str = "joint_right_arm_1_x4_1_dof_x4"
    hand: Node = RightHand()


class LeftLeg(Node):
    hip_roll: str = "joint_legs_1_x8_2_dof_x8"
    hip_yaw: str = "joint_legs_1_left_leg_1_x8_1_dof_x8"
    hip_pitch: str = "joint_legs_1_left_leg_1_x10_1_dof_x10"
    knee_motor: str = "joint_legs_1_left_leg_1_x10_2_dof_x10"
    knee: str = "joint_legs_1_left_leg_1_knee_revolute"
    ankle_motor: str = "joint_legs_1_left_leg_1_x6_1_dof_x6"
    ankle: str = "joint_legs_1_left_leg_1_ankle_revolute"
    foot_roll: str = "joint_legs_1_left_leg_1_x4_1_dof_x4"


class RightLeg(Node):
    hip_roll: str = "joint_legs_1_x8_1_dof_x8"
    hip_yaw: str = "joint_legs_1_right_leg_1_x8_1_dof_x8"
    hip_pitch: str = "joint_legs_1_right_leg_1_x10_2_dof_x10"
    knee_motor: str = "joint_legs_1_right_leg_1_x10_1_dof_x10"
    knee: str = "joint_legs_1_right_leg_1_knee_revolute"
    ankle_motor: str = "joint_legs_1_right_leg_1_x6_1_dof_x6"
    ankle: str = "joint_legs_1_right_leg_1_ankle_revolute"
    foot_roll: str = "joint_legs_1_right_leg_1_x4_1_dof_x4"


class Legs(Node):
    left: Node = LeftLeg()
    right: Node = RightLeg()


class Stompy(Node):
    head: Node = Head()
    torso: Node = Torso()
    left_arm: Node = LeftArm()
    right_arm: Node = RightArm()
    legs: Node = Legs()


def print_joints() -> None:
    joints = Stompy.all_joints()
    assert len(joints) == len(set(joints)), "Duplicate joint names found!"
    print(Stompy())


if __name__ == "__main__":
    # python -m sim.stompy.joints
    print_joints()
