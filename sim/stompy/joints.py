"""Defines a more Pythonic interface for specifying the joint names.

The best way to re-generate this snippet for a new robot is to use the
`sim/scripts/print_joints.py` script. This script will print out a hierarchical
tree of the various joint names in the robot.
"""

import textwrap
from abc import ABC
from typing import Dict, List, Union

import numpy as np


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
    left_right = "joint_head_1_x4_1_dof_x4"
    up_down = "joint_head_1_x4_2_dof_x4"


class Torso(Node):
    pitch = "joint_torso_1_x8_1_dof_x8"


class LeftHand(Node):
    hand_roll = "joint_left_arm_2_hand_1_x4_1_dof_x4"
    hand_grip = "joint_left_arm_2_hand_1_x4_2_dof_x4"
    slider_a = "joint_left_arm_2_hand_1_slider_1"
    slider_b = "joint_left_arm_2_hand_1_slider_2"


class LeftArm(Node):
    shoulder_yaw = "joint_left_arm_2_x8_1_dof_x8"
    shoulder_pitch = "joint_left_arm_2_x8_2_dof_x8"
    shoulder_roll = "joint_left_arm_2_x6_1_dof_x6"
    elbow_yaw = "joint_left_arm_2_x6_2_dof_x6"
    elbow_roll = "joint_left_arm_2_x4_1_dof_x4"
    hand = LeftHand()


class RightHand(Node):
    hand_roll = "joint_right_arm_1_hand_1_x4_1_dof_x4"
    hand_grip = "joint_right_arm_1_hand_1_x4_2_dof_x4"
    slider_a = "joint_right_arm_1_hand_1_slider_1"
    slider_b = "joint_right_arm_1_hand_1_slider_2"


class RightArm(Node):
    shoulder_yaw = "joint_right_arm_1_x8_1_dof_x8"
    shoulder_pitch = "joint_right_arm_1_x8_2_dof_x8"
    shoulder_roll = "joint_right_arm_1_x6_1_dof_x6"
    elbow_yaw = "joint_right_arm_1_x6_2_dof_x6"
    elbow_roll = "joint_right_arm_1_x4_1_dof_x4"
    hand = RightHand()


class LeftLeg(Node):
    hip_roll = "joint_legs_1_x8_2_dof_x8"
    hip_yaw = "joint_legs_1_left_leg_1_x8_1_dof_x8"
    hip_pitch = "joint_legs_1_left_leg_1_x10_1_dof_x10"
    knee_motor = "joint_legs_1_left_leg_1_x10_2_dof_x10"
    knee = "joint_legs_1_left_leg_1_knee_revolute"
    ankle_motor = "joint_legs_1_left_leg_1_x6_1_dof_x6"
    ankle = "joint_legs_1_left_leg_1_ankle_revolute"
    foot_roll = "joint_legs_1_left_leg_1_x4_1_dof_x4"


class RightLeg(Node):
    hip_roll = "joint_legs_1_x8_1_dof_x8"
    hip_yaw = "joint_legs_1_right_leg_1_x8_1_dof_x8"
    hip_pitch = "joint_legs_1_right_leg_1_x10_2_dof_x10"
    knee_motor = "joint_legs_1_right_leg_1_x10_1_dof_x10"
    knee = "joint_legs_1_right_leg_1_knee_revolute"
    ankle_motor = "joint_legs_1_right_leg_1_x6_1_dof_x6"
    ankle = "joint_legs_1_right_leg_1_ankle_revolute"
    foot_roll = "joint_legs_1_right_leg_1_x4_1_dof_x4"


class Legs(Node):
    left = LeftLeg()
    right = RightLeg()


class Stompy(Node):
    head = Head()
    torso = Torso()
    left_arm = LeftArm()
    right_arm = RightArm()
    legs = Legs()

    @classmethod
    def default_positions(cls) -> Dict[str, float]:
        return {
            Stompy.head.left_right: np.deg2rad(-54),
            Stompy.head.up_down: 0.0,
            Stompy.torso.pitch: 0.0,
            Stompy.left_arm.shoulder_yaw: np.deg2rad(60),
            Stompy.left_arm.shoulder_pitch: np.deg2rad(60),
            Stompy.right_arm.shoulder_yaw: np.deg2rad(-60),
        }


def print_joints() -> None:
    joints = Stompy.all_joints()
    assert len(joints) == len(set(joints)), "Duplicate joint names found!"
    print(Stompy())


if __name__ == "__main__":
    # python -m sim.stompy.joints
    print_joints()
