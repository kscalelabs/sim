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
        Left = "Hip_Pitch_Left"
        Right = "Hip_Pitch_Right"
    
    Lift_Joint_Limit_2 = "Hip_Lift_Joint_Limit_2"

class RightLeg(Node):
    hip_pitch = "Hip_Pitch_Right"
    hip_lift = "Hip_Lift_Joint_Limit_2"
    hip_roll = "Thigh_Rotate_2"
    knee_rotate = "Knee_Rotate_2"
    foot_rotate = "Foot_rotate_2"

class LeftLeg(Node):
    hip_pitch = "Hip_Pitch_Left"
    hip_lift = "Hip_Lift_Joint_Limit"
    hip_roll = "Thigh_Rotate"
    knee_rotate = "Knee_Rotate"
    foot_rotate = "Foot_rotate"

class Legs(Node):
    left = LeftLeg()
    right = RightLeg()

class Robot(Node):
    height = 0.021
    rotation = [0.5, -0.4996, -0.5, 0.5004]


    legs = Legs()

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
        }

    # p_gains
    @classmethod
    def stiffness(cls) -> Dict[str, float]:
        return {
            "hip pitch": 250,
            "hip lift": 150,
            "hip roll": 250,
            "knee rotate": 250,
            "foot rotate": 150,
        }

    # d_gains
    @classmethod
    def damping(cls) -> Dict[str, float]:
        return {
            "hip pitch": 10,
            "hip lift": 10,
            "hip roll": 10,
            "knee rotate": 10,
            "foot rotate": 10,
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
        }

    @classmethod
    def friction(cls) -> Dict[str, float]:
        return {
            "hip pitch": 0.0,
            "hip lift": 0.0,
            "hip roll": 0.0,
            "knee rotate": 0.0,
            "foot rotate": 0.0,
        }



def print_joints() -> None:
    joints = Robot.all_joints()
    assert len(joints) == len(set(joints)), "Duplicate joint names found!"
    print(Robot())

if __name__ == "__main__":
    print_joints()

