# mypy: disable-error-code="valid-newtype"
"""This script updates the URDF file to fix the joints of the robot."""

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

from sim.scripts.create_mjcf import load_embodiment


def update_urdf(model_path: str, embodiment: str) -> None:
    tree = ET.parse(Path(model_path) / "robot.urdf")
    root = tree.getroot()
    robot = load_embodiment(embodiment)
    print(robot.default_standing())
    revolute_joints = set(robot.default_standing().keys())

    joint_limits = robot.default_limits()
    effort = robot.effort()
    velocity = robot.velocity()
    friction = robot.friction()

    for joint in root.findall("joint"):
        joint_name = joint.get("name")

        if joint_name not in revolute_joints:
            joint.set("type", "fixed")
        else:
            limit = joint.find("limit")
            if limit is not None:
                limits = joint_limits.get(joint_name, {})
                if limits:
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
    tree.write(Path(model_path) / "robot_fixed.urdf", xml_declaration=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Update URDF file to fix robot joints.")
    parser.add_argument("--model_path", type=str, help="Path to the model directory", default="sim/resources/gpr")
    parser.add_argument("--embodiment", type=str, help="Embodiment to use", default="gpr")
    args = parser.parse_args()

    update_urdf(args.model_path, args.embodiment)
    # create_mjcf(Path(args.model_path) / "robot")


if __name__ == "__main__":
    main()
