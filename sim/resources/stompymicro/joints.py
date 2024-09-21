"""Defines a more Pythonic interface for specifying the joint names for stompymicro."""

## Print joints.py output:
# Hip
#   Pitch
#     Left: Hip_Pitch_Left
#     Right: Hip_Pitch_Right
#   Lift_Joint_Limit_2: Hip_Lift_Joint_Limit_2
# Foot_rotate_2: Foot_rotate_2
# Knee_Rotate_2: Knee_Rotate_2
# Thigh_Rotate_2: Thigh_Rotate_2


import textwrap
from abc import ABC
from typing import Dict, List, Tuple, Union

class Node(ABC):



class Hip(Node):
    class Pitch(Node):
        Left = "Hip_Pitch_Left"
        Right = "Hip_Pitch_Right"
    
    Lift_Joint_Limit_2 = "Hip_Lift_Joint_Limit_2"

class Robot(Node):
    hip = Hip()
    Foot_rotate_2 = "Foot_rotate_2"
    Knee_Rotate_2 = "Knee_Rotate_2"
    Thigh_Rotate_2 = "Thigh_Rotate_2"

    @classmethod
    def default_standing(cls) -> Dict[str, float]:
        return {
            Robot.hip.Pitch.Left: 0,
            Robot.hip.Pitch.Right: 0,
            Robot.hip.Lift_Joint_Limit_2: 0,
            Robot.Foot_rotate_2: 0,
            Robot.Knee_Rotate_2: 0,
            Robot.Thigh_Rotate_2: 0,
        }


def print_joints() -> None:
    joints = Robot.all_joints()
    assert len(joints) == len(set(joints)), "Duplicate joint names found!"
    print(Robot())

if __name__ == "__main__":
    print_joints()

