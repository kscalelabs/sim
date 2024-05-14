# mypy: disable-error-code="valid-newtype"
"""This script updates the MJCF version for proper rendering."""
import os
import xml.etree.ElementTree as ET

import mujoco


def create_mjcf():
    model_xml = mujoco.MjModel.from_xml_path(os.path.join(os.getenv("MODEL_DIR"), "robot_fixed.urdf"))
    mujoco.mj_saveLastXML(os.path.join(os.getenv("MODEL_DIR"), "robot.xml"), model_xml)

    # Add light and floor to the xml file and move all elements to a new root body
    tree = ET.parse(os.path.join(os.getenv("MODEL_DIR"), "robot.xml"))
    root = tree.getroot()
    # Create light element
    worldbody = root.find("worldbody")
    # Insert new elements at the beginning of the worldbody

    visual = ET.Element( "visual",{})
    visual.append(ET.Element("quality", {"shadowsize":"4096"}))
    visual.append(ET.Element("map", {"znear":"0.05"}))
    worldbody.insert(0, visual)

    option =  ET.Element("option", {})

    new_root_body = ET.Element("body", name="root", pos="-1.4 0 -0.3", quat="1 0 0 0")

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

    with open(os.path.join(os.getenv("MODEL_DIR"), "robot.xml"), "w") as f:
        f.write(modified_xml)
    # 5. https://github.com/roboterax/humanoid-gym/blob/main/humanoid/scripts/sim2sim.py
# 


if __name__ == "__main__":
    create_mjcf()
