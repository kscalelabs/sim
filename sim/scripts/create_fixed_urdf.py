# mypy: disable-error-code="valid-newtype"
"""This script updates the URDF file to fix the joints of the robot."""

import xml.etree.ElementTree as ET
from sim.stompy.joints import StompyFixed

STOMPY_URDF = "stompy/robot.urdf"
STOMPY_MJCF = "stompy/robot.xml"


def update_urdf():
    tree = ET.parse()
    root = tree.getroot()
    stompy = StompyFixed()

    revolute_joints = set(stompy.default_standing().keys())
    joint_limits = stompy.default_limits()

    for joint in root.findall("joint"):
        joint_name = joint.get("name")
        if joint_name not in revolute_joints:
            print(joint)
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
    tree.write("stompy/robot_fixed.urdf")


def update_mjcf():
    tree = ET.parse(STOMPY_MJCF)
    root = tree.getroot()
    # Create light element
    light = ET.Element("light", {
        "cutoff": "100",
        "diffuse": "1 1 1",
        "dir": "-0 0 -1.3",
        "directional": "true",
        "exponent": "1",
        "pos": "0 0 1.3",
        "specular": ".1 .1 .1"
    })

    # Create floor geometry element
    floor = ET.Element("geom", {
        "conaffinity": "1",
        "condim": "3",
        "name": "floor",
        "pos": "0 0 -1.5",
        "rgba": "0.8 0.9 0.8 1",
        "size": "40 40 40",
        "type": "plane",
        "material": "MatPlane"
    })
    # Access the <worldbody> element
    worldbody = root.find("worldbody")
    # Insert new elements at the beginning of the worldbody
    worldbody.insert(0, floor)
    worldbody.insert(0, light)

    new_root_body = ET.Element("body", name="root", pos="-1.4 0 -0.3", quat="1 0 0 1")

    # List to store items to be moved to the new root body
    items_to_move = []
    # Gather all children (geoms and bodies) that need to be moved under the new root body
    for element in worldbody:
        items_to_move.append(element)
    # Move gathered elements to the new root body
    for item in items_to_move:
        worldbody.remove(item)
        new_root_body.append(item)
    # Add the new root body to the worldbody
    worldbody.append(new_root_body)
    worldbody2 = worldbody
    # # Create a new root worldbody and add the new root body to it
    # new_worldbody = ET.Element("worldbody", name="new_root")
    index = list(root).index(worldbody)

    # # Remove the old <worldbody> and insert the new one at the same index
    root.remove(worldbody)
    root.insert(index, worldbody2)
    modified_xml = ET.tostring(root, encoding="unicode")

    with open("stompy/robot_fixed.xml", "w") as f:
        f.write(modified_xml)


if __name__ == "__main__":
    # update_urdf()
