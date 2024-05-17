# mypy: disable-error-code="operator,union-attr"
"""Defines common types and functions for exporting MJCF files.

API reference:
https://github.com/google-deepmind/mujoco/blob/main/src/xml/xml_native_writer.cc#L780

python sim.scripts/create_mjcf.py

Todo:
    -1. Inertial information
    0. IMU right position - base
    1. Add to all geoms
    2. Condim 3 and 4 and difference in results
    3.

"""
import xml.etree.ElementTree as ET
from pathlib import Path
import xml.dom.minidom

from kol.formats import mjcf
from sim.stompy.joints import StompyFixed


STOMPY_HEIGHT = 1.0


def _pretty_print_xml(xml_string):
    """Formats the provided XML string into a pretty-printed version."""
    parsed_xml = xml.dom.minidom.parseString(xml_string)
    return parsed_xml.toprettyxml(indent="  ")


class Sim2SimRobot(mjcf.Robot):
    """A class to adapt the world in a Mujoco XML file."""

    def adapt_world(self) -> None:
        root = self.tree.getroot()
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

        # Move gathered elements to the new root body
        for item in items_to_move:
            worldbody.remove(item)
            new_root_body.append(item)

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

        motors = []
        sensors = []
        for joint, limits in StompyFixed.default_limits().items():
            if joint in StompyFixed.default_standing().keys():
                motors.append(
                    mjcf.Motor(
                        name=joint, joint=joint, gear=1, ctrlrange=(limits["lower"], limits["upper"]), ctrllimited=True
                    )
                )
                sensors.extend(
                    [
                        mjcf.Actuatorpos(name=joint + "_p", actuator=joint, user="13"),
                        mjcf.Actuatorvel(name=joint + "_v", actuator=joint, user="13"),
                        mjcf.Actuatorfrc(name=joint + "_f", actuator=joint, user="13", noise=0.001),
                    ]
                )

        # Add motors and sensors
        root.append(mjcf.Actuator(motors).to_xml())
        root.append(mjcf.Sensor(sensors).to_xml())

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

        root.insert(1,
            mjcf.Option(
                timestep=0.001,
                viscosity=1e-6,
                iterations=50,
                solver="PGS",
                gravity=(0, 0, -9.81),
                flag=mjcf.Flag(frictionloss="enable"),
            ).to_xml()
        )

        # TODO - test the physical parameters
        root.insert(1,
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
                # visualgem?
                # joint param damping
            ).to_xml()
        )
        self.tree = ET.ElementTree(root)

    def save(self, path: str | Path) -> None:
        rough_string = ET.tostring(self.tree.getroot(), "utf-8")
        # Pretty print the XML
        formatted_xml = _pretty_print_xml(rough_string)

        with open(path, "w") as f:
            f.write(formatted_xml)


if __name__ == "__main__":
    robot_name = "robot_fixed"
    robot = Sim2SimRobot(
        robot_name,
        Path("/Users/pfb30/sim/stompy"),
        mjcf.Compiler(angle="radian", meshdir="meshes", autolimits=True),
        remove_inertia=False,
    )
    # TODO - test eulerseq setup
    # , eulerseq="xyz"))
    robot.adapt_world()
    robot.save(f"/Users/pfb30/sim/stompy/robot_new.xml")
