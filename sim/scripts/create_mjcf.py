# mypy: disable-error-code="operator,union-attr"
"""Defines common types and functions for exporting MJCF files.

Run:
    python sim/scripts/create_mjcf.py /path/to/stompy.xml

Todo:
    0. Add IMU to the right position
    1. Armature damping setup for different parts of body
    2. Test control range limits?
    3. Add inertia in the first part of the body

"""

import argparse
import logging
import xml.dom.minidom
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Union

from sim import mjcf
from sim.stompy.joints import Stompy

logger = logging.getLogger(__name__)

# Links that will have collision with the floor
COLLISION_LINKS = [
    "right_foot_1_rubber_grip_3_simple",
    "left_foot_1_rubber_grip_1_simple",
]

ROOT_HEIGHT = 0.72

stompy = Stompy()


def _pretty_print_xml(xml_string: str) -> str:
    """Formats the provided XML string into a pretty-printed version."""
    parsed_xml = xml.dom.minidom.parseString(xml_string)
    pretty_xml = parsed_xml.toprettyxml(indent="  ")

    # Split the pretty-printed XML into lines and filter out empty lines
    lines = pretty_xml.split("\n")
    non_empty_lines = [line for line in lines if line.strip() != ""]
    return "\n".join(non_empty_lines)


class Sim2SimRobot(mjcf.Robot):
    """A class to adapt the world in a Mujoco XML file."""

    def adapt_world(self, add_floor: bool = True, remove_frc_range: bool = False) -> None:
        root: ET.Element = self.tree.getroot()

        if add_floor:
            asset = root.find("asset")
            asset.append(
                ET.Element(
                    "texture",
                    name="texplane",
                    type="2d",
                    builtin="checker",
                    rgb1=".0 .0 .0",
                    rgb2=".8 .8 .8",
                    width="100",
                    height="108",
                )
            )
            asset.append(
                ET.Element(
                    "material",
                    name="matplane",
                    reflectance="0.",
                    texture="texplane",
                    texrepeat="1 1",
                    texuniform="true",
                )
            )
            asset.append(ET.Element("material", name="visualgeom", rgba="0.5 0.9 0.2 1"))

        compiler = root.find("compiler")
        if self.compiler is not None:
            compiler = self.compiler.to_xml(compiler)

        worldbody = root.find("worldbody")
        worldbody.insert(
            0,
            mjcf.Light(
                directional=True,
                diffuse=(0.4, 0.4, 0.4),
                specular=(0.1, 0.1, 0.1),
                pos=(0, 0, 5.0),
                dir=(0, 0, -1),
                castshadow=False,
            ).to_xml(),
        )
        worldbody.insert(
            0,
            mjcf.Light(
                directional=True, diffuse=(0.6, 0.6, 0.6), specular=(0.2, 0.2, 0.2), pos=(0, 0, 4), dir=(0, 0, -1)
            ).to_xml(),
        )
        if add_floor:
            worldbody.insert(
                0,
                mjcf.Geom(
                    name="ground",
                    type="plane",
                    size=(0, 0, 1),
                    pos=(0.001, 0, 0),
                    quat=(1, 0, 0, 0),
                    material="matplane",
                    condim=1,
                    conaffinity=15,
                ).to_xml(),
            )

        motors: List[mjcf.Motor] = []
        sensor_pos: List[mjcf.Actuatorpos] = []
        sensor_vel: List[mjcf.Actuatorvel] = []
        sensor_frc: List[mjcf.Actuatorfrc] = []
        # Create motors and sensors for the joints
        joints = list(root.findall("joint"))
        for joint, _ in stompy.default_limits().items():
            if joint in stompy.default_standing().keys():
                joint_name = joint
                limit = 200
                keys = stompy.effort().keys()
                for key in keys:
                    if key in joint_name:
                        limit = stompy.effort()[key]

                motors.append(
                    mjcf.Motor(
                        name=joint,
                        joint=joint,
                        gear=1,
                        ctrlrange=(-limit, limit),
                        ctrllimited=True,
                    )
                )
                sensor_pos.append(mjcf.Actuatorpos(name=joint + "_p", actuator=joint, user="13"))
                sensor_vel.append(mjcf.Actuatorvel(name=joint + "_v", actuator=joint, user="13"))
                sensor_frc.append(mjcf.Actuatorfrc(name=joint + "_f", actuator=joint, user="13", noise=0.001))

        root = self.update_joints(root)

        # Add motors and sensors
        root.append(mjcf.Actuator(motors).to_xml())
        root.append(mjcf.Sensor(sensor_pos, sensor_vel, sensor_frc).to_xml())

        # TODO: Add additional sensors when necessary
        sensors = root.find("sensor")
        sensors.extend(
            [
                ET.Element("framequat", name="orientation", objtype="site", noise="0.001", objname="imu"),
                ET.Element("gyro", name="angular-velocity", site="imu", noise="0.005", cutoff="34.9"),
                # ET.Element("framepos", name="position", objtype="site", noise="0.001", objname="imu"),
                # ET.Element("velocimeter", name="linear-velocity", site="imu", noise="0.001", cutoff="30"),
                # ET.Element("accelerometer", name="linear-acceleration", site="imu", noise="0.005", cutoff="157"),
                # ET.Element("magnetometer", name="magnetometer", site="imu"),
            ]
        )

        root.insert(
            1,
            mjcf.Option(
                timestep=0.001,
                iterations=50,
                solver="PGS",
                gravity=(0, 0, -9.81),
            ).to_xml(),
        )

        visual_geom = ET.Element("default", {"class": "visualgeom"})
        geom_attributes = {"material": "visualgeom", "condim": "1", "contype": "0", "conaffinity": "0"}
        ET.SubElement(visual_geom, "geom", geom_attributes)

        root.insert(
            1,
            mjcf.Default(
                joint=mjcf.Joint(armature=0.01, stiffness=0, damping=0.01, limited=True, frictionloss=0.01),
                motor=mjcf.Motor(ctrllimited=True),
                equality=mjcf.Equality(solref=(0.001, 2)),
                geom=mjcf.Geom(
                    solref=(0.001, 2),
                    friction=(0.9, 0.2, 0.2),
                    condim=4,
                    contype=1,
                    conaffinity=15,
                ),
                visual_geom=visual_geom,
            ).to_xml(),
        )

        # Locate actual root body inside of worldbody
        root_body = worldbody.find(".//body")
        # Make position and orientation of the root body
        root_body.set("pos", "0 0 0")
        root_body.set("quat", "1 0 0 0")
        root_body.insert(0, ET.Element("joint", name="floating_base_joint", type="free"))

        # Add cameras and imu
        root_body.insert(1, ET.Element("camera", name="front", pos="0 -3 1", xyaxes="1 0 0 0 1 2", mode="trackcom"))
        root_body.insert(
            2,
            ET.Element(
                "camera",
                name="side",
                pos="-2.893 -1.330 0.757",
                xyaxes="0.405 -0.914 0.000 0.419 0.186 0.889",
                mode="trackcom",
            ),
        )
        root_body.insert(3, ET.Element("site", name="imu", size="0.01", pos="0 0 0"))

        # add visual geom logic
        for body in root.findall(".//body"):
            original_geoms = list(body.findall("geom"))
            for geom in original_geoms:
                geom.set("class", "visualgeom")
                # Create a new geom element
                new_geom = ET.Element("geom")
                new_geom.set("type", geom.get("type"))
                new_geom.set("rgba", geom.get("rgba"))
                new_geom.set("mesh", geom.get("mesh"))
                if geom.get("pos"):
                    new_geom.set("pos", geom.get("pos"))
                if geom.get("quat"):
                    new_geom.set("quat", geom.get("quat"))
                # Exclude collision meshes
                if geom.get("mesh") not in COLLISION_LINKS:
                    new_geom.set("contype", "0")
                    new_geom.set("conaffinity", "0")
                    new_geom.set("group", "1")
                    new_geom.set("density", "0")

                # Append the new geom to the body
                index = list(body).index(geom)
                body.insert(index + 1, new_geom)

        if remove_frc_range:
            for body in root.findall(".//body"):
                joints = list(body.findall("joint"))
                for join in joints:
                    if "actuatorfrcrange" in join.attrib:
                        join.attrib.pop("actuatorfrcrange")

        # Adding keyframe
        default_standing = stompy.default_standing()
        qpos = [0, 0, ROOT_HEIGHT, 1, 0, 0, 0] + list(default_standing.values())
        key = mjcf.Key(name="default", qpos=" ".join(map(str, qpos)))
        keyframe = mjcf.Keyframe(keys=[key])
        root.append(keyframe.to_xml())

        # Swap left and right leg
        parent_body = root.find(".//body[@name='root']")
        left = parent_body.find(".//body[@name='link_leg_assembly_left_1_rmd_x12_150_mock_1_inner_x12_150_1']")
        right = parent_body.find(".//body[@name='link_leg_assembly_right_1_rmd_x12_150_mock_1_inner_x12_150_1']")
        if left is not None and right is not None:
            left_index = list(parent_body).index(left)
            right_index = list(parent_body).index(right)
            # Swap the bodies
            parent_body[left_index], parent_body[right_index] = parent_body[right_index], parent_body[left_index]

    def update_joints(self, root: ET.Element) -> None:
        joint_limits = stompy.default_limits()

        for joint in root.findall(".//joint"):
            joint_name = joint.get("name")
            if joint_name in joint_limits:
                limits = joint_limits.get(joint_name)
                lower = str(limits.get("lower", 0.0))
                upper = str(limits.get("upper", 0.0))
                joint.set("range", f"{lower} {upper}")

                damping = 0.01
                keys = stompy.damping().keys()
                for key in keys:
                    if key in joint_name:
                        damping = stompy.damping()[key]
                joint.set("damping", str(damping))

                stiffness = 0
                keys = stompy.stiffness().keys()
                for key in keys:
                    if key in joint_name:
                        stiffness = stompy.stiffness()[key]

                joint.set("stiffness", str(stiffness))

        return root

    def save(self, path: Union[str, Path]) -> None:
        rough_string = ET.tostring(self.tree.getroot(), "utf-8")
        # Pretty print the XML
        formatted_xml = _pretty_print_xml(rough_string)
        logger.info("XML:\n%s", formatted_xml)
        with open(path, "w") as f:
            f.write(formatted_xml)


def create_mjcf(filepath: str) -> None:
    """Create a MJCF file for the Stompy robot."""
    path = Path(filepath)
    robot_name = path.stem
    path = path.parent
    robot = Sim2SimRobot(
        robot_name,
        path,
        mjcf.Compiler(angle="radian", meshdir="meshes", autolimits=True, eulerseq="zyx"),
    )
    robot.adapt_world()
    robot.save(path / f"{robot_name}_fixed.xml")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a MJCF file for the Stompy robot.")
    parser.add_argument("filepath", type=str, help="The path to load and save the MJCF file.")
    args = parser.parse_args()
    # Robot name is whatever string comes right before ".urdf" extension
    create_mjcf(args.filepath)
