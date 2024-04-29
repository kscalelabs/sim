# mypy: disable-error-code="valid-newtype"
"""This script updates the MJCF version for proper rendering.

The mcf format can be created by loading URDF file to the viewer and saving as xml.
"""
import xml.etree.ElementTree as ET

STOMPY_MJCF = "stompy/robot.xml"


def update_mjcf():
    tree = ET.parse(STOMPY_MJCF)
    root = tree.getroot()
    # Create light element
    light = ET.Element(
        "light",
        {
            "cutoff": "100",
            "diffuse": "1 1 1",
            "dir": "-0 0 -1.3",
            "directional": "true",
            "exponent": "1",
            "pos": "0 0 1.3",
            "specular": ".1 .1 .1",
        },
    )

    # Create floor geometry element
    floor = ET.Element(
        "geom",
        {
            "conaffinity": "1",
            "condim": "3",
            "name": "floor",
            "pos": "0 0 -1.5",
            "rgba": "0.8 0.9 0.8 1",
            "size": "40 40 40",
            "type": "plane",
            "material": "MatPlane",
        },
    )
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

    with open(STOMPY_MJCF, "w") as f:
        f.write(modified_xml)


if __name__ == "__main__":
    # update_urdf()
    update_mjcf()
