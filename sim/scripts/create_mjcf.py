# mypy: disable-error-code="operator,union-attr"
"""Defines common types and functions for exporting MJCF files.

Run:
    python sim/scripts/create_mjcf.py

Todo:
    0. Add IMU right position - base
    1. Armature damping setup for different parts of body
    2. Control range limits? check
    3. NO inertia in the first part of the body
    4. torso_1_top_torso_1 - no inertia there!
"""

import logging
import xml.dom.minidom
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Union

from sim.scripts import mjcf

from sim.env import stompy_mjcf_path
from sim.stompy.joints import StompyFixed

logger = logging.getLogger(__name__)

STOMPY_HEIGHT = 1.0


def _pretty_print_xml(xml_string: str) -> str:
    """Formats the provided XML string into a pretty-printed version."""
    parsed_xml = xml.dom.minidom.parseString(xml_string)
    return parsed_xml.toprettyxml(indent="  ")


class Sim2SimRobot(mjcf.Robot):
    """A class to adapt the world in a Mujoco XML file."""

    def adapt_world(self) -> None:
        root: ET.Element = self.tree.getroot()

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
                "material", name="matplane", reflectance="0.", texture="texplane", texrepeat="1 1", texuniform="true"
            )
        )
        asset.append(
            ET.Element(
                "material", name="visualgeom", rgba="0.5 0.9 0.2 1"
            )
        )

        compiler = root.find("compiler")
        if self.compiler is not None:
            compiler = self.compiler.to_xml(compiler)

        worldbody = root.find("worldbody")
        # List to store items to be moved to the new root body
        items_to_move = []
        # Gather all children (geoms and bodies) that need to be moved under the new root body
        for element in worldbody:
            items_to_move.append(element)

        new_root_body = mjcf.Body(name="root", pos=(0, 0, STOMPY_HEIGHT), quat=(1, 0, 0, 0)).to_xml()
        # Add joints to all the movement of the base
        new_root_body.extend(
            [
                mjcf.Joint(name="root_x", type="slide", axis=(1, 0, 0), limited=False).to_xml(),
                mjcf.Joint(name="root_y", type="slide", axis=(0, 1, 0), limited=False).to_xml(),
                mjcf.Joint(name="root_z", type="slide", axis=(0, 0, 1), limited=False).to_xml(),
                mjcf.Joint(name="root_ball", type="ball", limited=False).to_xml(),
            ]
        )
        

        # Add imu site to the body - relative position to the body
        # check at what stage we use this
        new_root_body.append(mjcf.Site(name="imu", size=0.01, pos=(0, 0, 0)).to_xml())

        # Add the new root body to the worldbody
        worldbody.append(new_root_body)
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
        for joint, limits in StompyFixed.default_limits().items():
            if joint in StompyFixed.default_standing().keys():
                motors.append(
                    mjcf.Motor(
                        # name=joint, joint=joint, gear=1, ctrlrange=(limits["lower"], limits["upper"]), ctrllimited=True
                        name=joint, joint=joint, gear=1, ctrlrange=(-200, 200), ctrllimited=True
                    )
                )
                sensor_pos.append(mjcf.Actuatorpos(name=joint + "_p", actuator=joint, user="13"))
                sensor_vel.append(mjcf.Actuatorvel(name=joint + "_v", actuator=joint, user="13"))
                sensor_frc.append(mjcf.Actuatorfrc(name=joint + "_f", actuator=joint, user="13", noise=0.001))

        # Add motors and sensors
        root.append(mjcf.Actuator(motors).to_xml())
        root.append(mjcf.Sensor(sensor_pos, sensor_vel, sensor_frc).to_xml())

        # Add imus
        sensors = root.find("sensor")
        sensors.extend(
            [
                # TODO - pfb30 test that
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
                viscosity=1e-6,
                iterations=50,
                solver="PGS",
                gravity=(0, 0, -9.81),
                flag=mjcf.Flag(frictionloss="enable"),
            ).to_xml(),
        )

        visual_geom = ET.Element("default", {"class":"visualgeom"})
        geom_attributes = {
                'material': 'visualgeom',
                'condim': '1',
                'contype': '0',
                'conaffinity': '0'
            }
        ET.SubElement(visual_geom, 'geom', geom_attributes)

        root.insert(
            1,
            mjcf.Default(
                joint=mjcf.Joint(armature=0.01, damping=0.1, limited=True, frictionloss=0.01),
                motor=mjcf.Motor(ctrllimited=True),
                equality=mjcf.Equality(solref=(0.001, 2)),
                geom=mjcf.Geom(
                    solref=(0.001, 2),
                    friction=(0.9, 0.2, 0.2),
                    condim=4,
                    contype=1,
                    conaffinity=15,
                ),
                visual_geom=visual_geom
                # TODO and joint param damping
            ).to_xml(),
        )


        # Move gathered elements to the new root body
        for item in items_to_move:
            worldbody.remove(item)
            new_root_body.append(item)
    
        self.tree = ET.ElementTree(root)

        root = self.tree.getroot()

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
                else:
                    print(geom)
                if geom.get("quat"):
                    new_geom.set("quat", geom.get("quat"))
                new_geom.set("contype", "0")
                new_geom.set("conaffinity", "0")
                new_geom.set("group", "1")
                new_geom.set("density", "0")
                
                # Append the new geom to the body
                index = list(body).index(geom)
                body.insert(index + 1, new_geom)
        

    def save(self, path: Union[str, Path]) -> None:
        rough_string = ET.tostring(self.tree.getroot(), "utf-8")
        # Pretty print the XML
        formatted_xml = _pretty_print_xml(rough_string)
        logger.info("XML:\n%s", formatted_xml)
        with open(path, "w") as f:
            f.write(formatted_xml)


if __name__ == "__main__":
    robot_name = "robot_fixed"
    robot = Sim2SimRobot(
        robot_name,
        # TODO - fix that
        Path("stompy"),
        mjcf.Compiler(angle="radian", meshdir="meshes", autolimits=True),
        remove_inertia=False,
    )

    robot.adapt_world()
    robot.save(stompy_mjcf_path(legs_only=True))
