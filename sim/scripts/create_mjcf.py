# mypy: disable-error-code="operator,union-attr"
"""Defines common types and functions for exporting MJCF files.

Run:
    python sim/scripts/create_mjcf.py /path/to/robot.xml

Todo:
    1. Add IMU to the right position
    2. Armature damping setup for different parts of body
"""

import argparse
import importlib
import logging
import os
import xml.dom.minidom
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, List, OrderedDict, Union

from sim.scripts import mjcf

logger = logging.getLogger(__name__)

DAMPING_DEFAULT = 0.02


def load_embodiment(embodiment: str) -> Any:
    # Dynamically import embodiment based on MODEL_DIR
    module_name = f"sim.resources.{embodiment}.joints"
    module = importlib.import_module(module_name)
    robot = getattr(module, "Robot")
    print(robot)
    return robot


def load_config() -> Any:
    # Dynamically import config based on MODEL_DIR
    model_dir = os.environ.get("MODEL_DIR", "stompypro")
    if "sim/" in model_dir:
        model_dir = model_dir.split("sim/")[1]
    model_dir = "stompypro"
    module_name = f"sim.{model_dir}.config"
    module = importlib.import_module(module_name)
    config = getattr(module, "Config")
    return config


def _pretty_print_xml(xml_string: str) -> str:
    """Formats the provided XML string into a pretty-printed version."""
    parsed_xml = xml.dom.minidom.parseString(xml_string)
    pretty_xml = parsed_xml.toprettyxml(indent="  ")

    # Split the pretty-printed XML into lines and filter out empty lines
    lines = pretty_xml.split("\n")
    non_empty_lines = [line for line in lines if line.strip() != ""]
    # Remove declaration
    return "\n".join(non_empty_lines[1:])


class Sim2SimRobot(mjcf.Robot):
    """A class to adapt the world in a Mujoco XML file."""

    def update_joints(self, root: ET.Element, damping: float = DAMPING_DEFAULT) -> ET.Element:
        joint_limits = robot.default_limits()
        default_standing = robot.default_standing()

        for body in root.findall(".//body"):
            joints_to_remove = []
            for joint in body.findall("joint"):
                joint_name = joint.get("name")
                if joint_name in joint_limits:
                    limits = joint_limits.get(joint_name)
                    lower = str(limits.get("lower", 0.0))
                    upper = str(limits.get("upper", 0.0))
                    joint.set("range", f"{lower} {upper}")

                # Comment to use Mujoco defaults
                # keys = robot.damping().keys()
                # for key in keys:
                #     if key in joint_name:
                #         joint_damping = damping
                #         joint.set("damping", str(joint_damping))
                #         print(f"Damping for {joint_name}: {joint_damping}")

                # keys = robot.stiffness().keys()
                # for key in keys:
                #     if key in joint_name:
                #         stiffness = robot.stiffness()[key]
                #         stiffness = 0.1
                #         joint.set("stiffness", str(stiffness))
                #         print(f"Stiffness for {joint_name}: {stiffness}")

                # Check if the joint is not in default_standing
                if joint_name not in default_standing:
                    joints_to_remove.append(joint)

            # Remove joints not in default_standing
            for joint in joints_to_remove:
                body.remove(joint)
                print(f"Removed joint: {joint.get('name')}")

        return root

    def adapt_world(self, add_floor: bool = True, remove_frc_range: bool = True) -> None:
        root: ET.Element = self.tree.getroot()

        worldbody = root.find("worldbody")
        new_root_body = mjcf.Body(name="root", pos=(0, 0, 1), quat=(1, 0, 0, 0)).to_xml()
        # add freejoint to root
        freejoint = ET.Element("freejoint", name="root")
        new_root_body.insert(0, freejoint)
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
        worldbody = root.find("worldbody")

        motors: List[mjcf.Motor] = []
        sensor_pos: List[mjcf.Actuatorpos] = []
        sensor_vel: List[mjcf.Actuatorvel] = []
        sensor_frc: List[mjcf.Actuatorfrc] = []
        # Create motors and sensors for the joints
        joints = list(root.findall(".//joint"))
        original_joints = joints.copy()
        for joint_xml in joints:  # robot.all_joints():
            joint = joint_xml.get("name")
            if joint in robot.default_standing().keys():
                joint_name = joint
                limit = 1000.0  # 200.0  # Ensure limit is a float
                keys = robot.effort().keys()
                for key in keys:
                    if key in joint_name:
                        limit = robot.effort()[key]
                print(f"Joint: {joint}, limit: {limit}")
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
            else:
                print(f"Joint: {joint} not in default_standing")
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
                joint=mjcf.Joint(armature=0.01, damping=0.01, limited=True, frictionloss=0.01),
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
        root_body.set("pos", "0 0 0")
        root_body.set("quat", " ".join(map(str, robot.rotation)))

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
                new_geom.set("type", geom.get("type") or "")  # Ensure type is not None
                new_geom.set("rgba", geom.get("rgba") or "1 0.5 0.75 1")  # Ensure rgba is not None

                # Check if geom has mesh or is a box
                if geom.get("mesh") is None:
                    if geom.get("type") == "box":
                        new_geom.set("type", "box")
                        new_geom.set("size", geom.get("size") or "")
                    else:
                        print(f"Unknown geom type: {geom.get('type')}")
                else:
                    new_geom.set("mesh", geom.get("mesh"))
                if geom.get("pos"):
                    new_geom.set("pos", geom.get("pos") or "")
                if geom.get("quat"):
                    new_geom.set("quat", geom.get("quat") or "")
                try:
                    # Exclude collision meshes
                    if geom.get("mesh") not in robot.collision_links:
                        new_geom.set("contype", "0")
                        new_geom.set("conaffinity", "0")
                        new_geom.set("group", "1")
                        new_geom.set("density", "0")
                except Exception as e:
                    print(e)

                # Append the new geom to the body
                index = list(body).index(geom)
                body.insert(index + 1, new_geom)

        for body in root.findall(".//body"):
            joints = list(body.findall("joint"))
            for join in joints:
                if "actuatorfrcrange" in join.attrib:
                    join.attrib.pop("actuatorfrcrange")

        default_standing = robot.default_standing()
        joint_defaults = list(default_standing.values())  # + [0.0] * (len(original_joints) - len(default_standing))
        # Flip the quaternion angle
        qpos = (
            [0, 0, robot.height]
            + [robot.rotation[3], robot.rotation[0], robot.rotation[1], robot.rotation[2]]
            + joint_defaults
        )

        # qpos = list(default_standing.values())

        # Pad qpos with zeros to match the number of joints
        # qpos += [0.0] * (len(original_joints) - len(qpos))
        # qpos = joint_defaults

        default_key = mjcf.Key(name="default", qpos=" ".join(map(str, qpos)))
        keyframe = mjcf.Keyframe(keys=[default_key])
        root.append(keyframe.to_xml())

        # Swap left and right leg since our setup - IDK if this is necessary (wesley)
        # parent_body = root.find(".//body[@name='root']")
        # if parent_body is not None:
        #     left = parent_body.find(".//body[@name='L_clav']")
        #     right = parent_body.find(".//body[@name='R_clav']")
        #     print(left.get("name"))
        #     print(right.get("name"))
        #     if left is not None and right is not None:
        #         left_index = list(parent_body).index(left)
        #         right_index = list(parent_body).index(right)
        #         # Swap the bodies
        #         parent_body[left_index], parent_body[right_index] = parent_body[right_index], parent_body[left_index]

        # # Remove the root body in the end
        # root_body = worldbody.find("./body[@name='root']")
        # children = list(root_body)
        # worldbody.remove(root_body)
        # for child in children:
        #     worldbody.append(child)

    def save(self, path: Union[str, Path]) -> None:
        rough_string = ET.tostring(self.tree.getroot(), "utf-8", xml_declaration=False)
        # Pretty print the XML
        formatted_xml = _pretty_print_xml(rough_string)
        logger.info("XML:\n%s", formatted_xml)
        with open(path, "w") as f:
            f.write(formatted_xml)


def create_mjcf(filepath: Path) -> None:
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
    parser = argparse.ArgumentParser(description="Create a MJCF file for the robot.")
    parser.add_argument("filepath", type=str, help="The path to load and save the MJCF file.")
    parser.add_argument("--robot", type=str, help="The robot name to load.")
    args = parser.parse_args()
    # Robot name is whatever string comes right before ".urdf" extension
    robot_name = args.filepath.split("/")[-1].split(".")[0]

    # Load the robot and config
    robot = load_embodiment(args.robot)
    create_mjcf(args.filepath)
