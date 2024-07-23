# mypy: disable-error-code="valid-newtype"
"""This script updates the URDF file to fix the joints of the robot."""

import xml.etree.ElementTree as ET
from pathlib import Path

from sim import mjcf
from sim.scripts.create_mjcf import Sim2SimRobot
from sim.stompy_legs.joints import Stompy

STOMPY_URDF = "sim/stompy_legs/robot.urdf"


def update_urdf() -> None:
    tree = ET.parse(STOMPY_URDF)
    root = tree.getroot()
    stompy = Stompy()
    print(stompy.default_standing())
    revolute_joints = set(stompy.default_standing().keys())
    joint_limits = stompy.default_limits()
    for joint in root.findall("joint"):
        joint_name = joint.get("name")
        if joint_name not in revolute_joints:
            joint.set("type", "fixed")
        else:
            limit = joint.find("limit")
            if limit is not None:
                limits = joint_limits.get(joint_name, {})
                lower = str(limits.get("lower", 0.0))
                upper = str(limits.get("upper", 0.0))
                limit.set("lower", lower)
                limit.set("upper", upper)
    # Save the modified URDF to a new file
    tree.write("sim/stompy_legs/robot_fixed.urdf")


if __name__ == "__main__":
    update_urdf()

    path = Path("sim/stompy_legs/robot.urdf")
    robot_name = path.stem
    path = path.parent
    robot = Sim2SimRobot(
        robot_name,
        path,
        mjcf.Compiler(angle="radian", meshdir="meshes", autolimits=True, eulerseq="zyx"),
    )
    robot.adapt_world()
    robot.save("sim/stompy_legs/robot_fixed.xml")
