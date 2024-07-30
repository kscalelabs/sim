# mypy: disable-error-code="valid-newtype"
"""This script updates the URDF file to fix the joints of the robot."""

import xml.etree.ElementTree as ET

from sim.scripts.create_mjcf import create_mjcf
from sim.stompy_legs.joints import Stompy

MODEL_PATH = "sim/stompy_legs"


def update_urdf() -> None:
    tree = ET.parse(MODEL_PATH + "/robot.urdf")
    root = tree.getroot()
    stompy = Stompy()
    print(stompy.default_standing())
    revolute_joints = set(stompy.default_standing().keys())
    joint_limits = stompy.default_limits()
    effort = stompy.effort()
    velocity = stompy.velocity()
    friction = stompy.friction()

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

                for key, value in effort.items():
                    if key in joint_name:
                        limit.set("effort", str(value))

                for key, value in velocity.items():
                    if key in joint_name:
                        limit.set("velocity", str(value))

            dynamics = joint.find("dynamics")
            if dynamics is not None:
                for key, value in friction.items():
                    if key in joint_name:
                        dynamics.set("friction", str(value))

    # Save the modified URDF to a new file
    tree.write(MODEL_PATH + "/robot_fixed.urdf", xml_declaration=False)


if __name__ == "__main__":
    update_urdf()
    create_mjcf(MODEL_PATH + "/robot")
