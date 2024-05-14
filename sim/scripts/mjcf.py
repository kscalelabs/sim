# mypy: disable-error-code="operator,union-attr"
"""Defines common types and functions for exporting MJCF files.

API reference:
https://github.com/google-deepmind/mujoco/blob/main/src/xml/xml_native_writer.cc#L780

python sim.scripts/create_mjcf.py
"""

import glob
import os
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import mujoco

from sim.stompy.joints import StompyFixed


@dataclass
class Compiler:
    coordinate: Literal["local", "global"] | None = None
    angle: Literal["radian", "degree"] = "radian"
    meshdir: str | None = None
    eulerseq: Literal["xyz", "zxy", "zyx", "yxz", "yzx", "xzy"] | None = None

    def to_xml(self, compiler: ET.Element | None = None) -> ET.Element:
        if compiler is None:
            compiler = ET.Element("compiler")
        if self.coordinate is not None:
            compiler.set("coordinate", self.coordinate)
        compiler.set("angle", self.angle)
        if self.meshdir is not None:
            compiler.set("meshdir", self.meshdir)
        if self.eulerseq is not None:
            compiler.set("eulerseq", self.eulerseq)
        return compiler


@dataclass
class Mesh:
    name: str
    file: str
    scale: tuple[float, float, float] | None = None

    def to_xml(self, root: ET.Element | None = None) -> ET.Element:
        if root is None:
            mesh = ET.Element("mesh")
        else:
            mesh = ET.SubElement(root, "mesh")
        mesh.set("name", self.name)
        mesh.set("file", self.file)
        if self.scale is not None:
            mesh.set("scale", " ".join(map(str, self.scale)))
        return mesh


@dataclass
class Joint:
    name: str | None = None
    type: Literal["hinge", "slide", "ball", "free"] | None = None
    pos: tuple[float, float, float] | None = None
    axis: tuple[float, float, float] | None = None
    limited: bool | None = None
    range: tuple[float, float] | None = None
    damping: float | None = None
    stiffness: float | None = None

    def to_xml(self, root: ET.Element | None = None) -> ET.Element:
        if root is None:
            joint = ET.Element("joint")
        else:
            joint = ET.SubElement(root, "joint")
        if self.name is not None:
            joint.set("name", self.name)
        if self.type is not None:
            joint.set("type", self.type)
        if self.pos is not None:
            joint.set("pos", " ".join(map(str, self.pos)))
        if self.axis is not None:
            joint.set("axis", " ".join(map(str, self.axis)))
        if self.range is not None:
            self.limited = True
            joint.set("range", " ".join(map(str, self.range)))
        else:
            self.limited = False
        joint.set("limited", str(self.limited))
        if self.damping is not None:
            joint.set("damping", str(self.damping))
        if self.stiffness is not None:
            joint.set("stiffness", str(self.stiffness))
        return joint


@dataclass
class Geom:
    mesh: str | None = None
    type: Literal["plane", "sphere", "cylinder", "box", "capsule", "ellipsoid", "mesh"] | None = None
    # size: float
    rgba: tuple[float, float, float, float] | None = None
    pos: tuple[float, float, float] | None = None
    quat: tuple[float, float, float, float] | None = None
    matplane: str | None = None
    material: str | None = None
    condim: int | None = None
    contype: int | None = None
    conaffinity: int | None = None

    def to_xml(self, root: ET.Element | None = None) -> ET.Element:
        if root is None:
            geom = ET.Element("geom")
        else:
            geom = ET.SubElement(root, "geom")
        if self.mesh is not None:
            geom.set("mesh", self.mesh)
        if self.type is not None:
            geom.set("type", self.type)
        if self.rgba is not None:
            geom.set("rgba", " ".join(map(str, self.rgba)))
        if self.pos is not None:
            geom.set("pos", " ".join(map(str, self.pos)))
        if self.quat is not None:
            geom.set("quat", " ".join(map(str, self.quat)))
        if self.matplane is not None:
            geom.set("matplane", self.matplane)
        if self.material is not None:
            geom.set("material", self.material)
        if self.condim is not None:
            geom.set("condim", str(self.condim))
        if self.contype is not None:
            geom.set("contype", str(self.contype))
        if self.conaffinity is not None:
            geom.set("conaffinity", str(self.conaffinity))
        return geom


@dataclass
class Body:
    name: str
    pos: tuple[float, float, float] | None = field(default=None)
    quat: tuple[float, float, float, float] | None = field(default=None)
    geom: Geom | None = field(default=None)
    joint: Joint | None = field(default=None)
    # TODO - fix inertia, until then rely on Mujoco's engine
    # inertial: Inertial = None

    def to_xml(self, root: ET.Element | None = None) -> ET.Element:
        if root is None:
            body = ET.Element("body")
        else:
            body = ET.SubElement(root, "body")
        body.set("name", self.name)
        if self.pos is not None:
            body.set("pos", " ".join(map(str, self.pos)))
        if self.quat is not None:
            body.set("quat", " ".join(f"{x:.8g}" for x in self.quat))
        if self.joint is not None:
            self.joint.to_xml(body)
        if self.geom is not None:
            self.geom.to_xml(body)
        return body


@dataclass
class Flag:
    sensornoise: str | None = None
    frictionloss: str | None = None

# @dataclass
# class Visual:
#     quality: 

@dataclass
class Option:
    timestep: float | None = None
    viscosity: float | None = None
    iteration: int | None = None
    solver: Literal["PGS", "CG", "Newton"] | None = None
    gravity: tuple[float, float, float] | None = None
    flag: Flag | None = None

    def to_xml(self, root: ET.Element | None = None) -> ET.Element:
        if root is None:
            option = ET.Element("option")
        else:
            option = ET.SubElement(root, "option")
        if self.iteration is not None:
            option.set("iteration", str(self.iteration))
        if self.timestep is not None:
            option.set("timestep", str(self.timestep))
        if self.viscosity is not None:
            option.set("viscosity", str(self.viscosity))
        if self.solver is not None:
            option.set("solver", self.solver)
        if self.gravity is not None:
            option.set("gravity", " ".join(map(str, self.gravity)))
        if self.flag is not None:
            self.flag.to_xml(option)
        return option


@dataclass
class Motor:
    name: str | None = None
    joint: str | None = None
    ctrlrange: tuple[float, float] | None = None
    ctrllimited: bool | None = None
    gear: float | None = None

    def to_xml(self, root: ET.Element | None = None) -> ET.Element:
        if root is None:
            motor = ET.Element("motor")
        else:
            motor = ET.SubElement(root, "motor")
        if self.name is not None:
            motor.set("name", self.name)
        if self.joint is not None:
            motor.set("joint", self.joint)
        if self.ctrllimited is not None:
            motor.set("ctrllimited", str(self.ctrllimited))
        if self.ctrlrange is not None:
            motor.set("ctrlrange", " ".join(map(str, self.ctrlrange)))
        if self.gear is not None:
            motor.set("gear", str(self.gear))
        return motor


@dataclass
class Light:
    directional: bool = True
    diffuse: tuple[float, float, float] | None = None
    specular: tuple[float, float, float] | None = None
    pos: tuple[float, float, float]  | None = None
    dir: tuple[float, float, float]  | None = None
    castshadow: bool | None = None

    def to_xml(self, root: ET.Element | None = None) -> ET.Element:
        if root is None:
            light = ET.Element("light")
        else:
            light = ET.SubElement(root, "light")
        light.set("directional", str(self.directional))
        if self.diffuse is not None:
            light.set("diffuse", " ".join(map(str, self.diffuse)))
        if self.specular is not None:
            light.set("specular", " ".join(map(str, self.specular)))
        if self.pos is not None:
            light.set("pos", " ".join(map(str, self.pos)))
        if self.dir is not None:
            light.set("dir", " ".join(map(str, self.dir)))
        if self.castshadow is not None:
            light.set("castshadow", str(self.castshadow))
        return light


@dataclass
class Equality:
    solref: tuple[float, float]

    def to_xml(self, root: ET.Element | None = None) -> ET.Element:
        if root is None:
            equality = ET.Element("equality")
        else:
            equality = ET.SubElement(root, "equality")
        equality.set("solref", " ".join(map(str, self.solref)))
        return equality


@dataclass
class Default:
    joint: Joint | None = None
    geom: Geom | None = None
    class_: str | None = None
    motor: Motor | None = None
    equality: Equality  | None = None

    def to_xml(self, default: ET.Element | None = None) -> ET.Element:
        if default is None:
            default = ET.Element("default")
        else:
            default = ET.SubElement(default, "default")
        if self.joint is not None:
            self.joint.to_xml(default)
        if self.geom is not None:
            self.geom.to_xml(default)
        if self.class_ is not None:
            default.set("class", self.class_)
        if self.motor is not None:
            self.motor.to_xml(default)
        if self.equality is not None:
            self.equality.to_xml(default)
        return default


@dataclass
class Actuator:
    motors: list[Motor]

    def to_xml(self, root: ET.Element | None = None) -> ET.Element:
        if root is None:
            actuator = ET.Element("actuator")
        else:
            actuator = ET.SubElement(root, "actuator")
        for motor in self.motors:
            motor.to_xml(actuator)
        return actuator


@dataclass
class Actuatorpos:
    name: str | None = None
    actuator: str | None = None
    user: str | None = None

    def to_xml(self, root: ET.Element | None = None) -> ET.Element:
        if root is None:
            actuatorpos = ET.Element("actuatorpos")
        else:
            actuatorpos = ET.SubElement(root, "actuatorpos")
        if self.name is not None:
            actuatorpos.set("name", self.name)
        if self.actuator is not None:
            actuatorpos.set("actuator", self.actuator)
        if self.user is not None:
            actuatorpos.set("user", self.user)
        return actuatorpos


@dataclass
class Actuatorvel:
    name: str | None = None
    actuator: str | None = None
    user: str | None = None

    def to_xml(self, root: ET.Element | None = None) -> ET.Element:
        if root is None:
            actuatorvel = ET.Element("actuatorvel")
        else:
            actuatorvel = ET.SubElement(root, "actuatorvel")
        if self.name is not None:
            actuatorvel.set("name", self.name)
        if self.actuator is not None:
            actuatorvel.set("actuator", self.actuator)
        if self.user is not None:
            actuatorvel.set("user", self.user)
        return actuatorvel


@dataclass
class Actuatorfrc:
    name: str | None = None
    actuator: str | None = None
    user: str | None = None
    noise: float | None = None

    def to_xml(self, root: ET.Element | None = None) -> ET.Element:
        if root is None:
            actuatorfrc = ET.Element("actuatorfrc")
        else:
            actuatorfrc = ET.SubElement(root, "actuatorfrc")
        if self.name is not None:
            actuatorfrc.set("name", self.name)
        if self.actuator is not None:
            actuatorfrc.set("actuator", self.actuator)
        if self.user is not None:
            actuatorfrc.set("user", self.user)
        if self.noise is not None:
            actuatorfrc.set("noise", str(self.noise))
        return actuatorfrc


@dataclass
class Sensor:
    actuatorpos: list[Actuatorpos] | None = None
    actuatorvel: list[Actuatorvel] | None = None
    actuatorfrc: list[Actuatorvel] | None = None

    def to_xml(self, root: ET.Element | None = None) -> ET.Element:
        if root is None:
            sensor = ET.Element("sensor")
        else:
            sensor = ET.SubElement(root, "sensor")
        if self.actuatorpos is not None:
            for actuatorpos in self.actuatorpos:
                actuatorpos.to_xml(sensor)
        if self.actuatorvel is not None:
            for actuatorvel in self.actuatorvel:
                actuatorvel.to_xml(sensor)
        if self.actuatorfrc is not None:
            for actuatorfrc in self.actuatorfrc:
                actuatorfrc.to_xml(sensor)
        return sensor


def _copy_stl_files(source_directory: str | Path, destination_directory: str | Path) -> None:
    # Ensure the destination directory exists, create if not
    os.makedirs(destination_directory, exist_ok=True)

    # Use glob to find all .stl files in the source directory
    pattern = os.path.join(source_directory, "*.stl")
    for file_path in glob.glob(pattern):
        destination_path = os.path.join(destination_directory, os.path.basename(file_path))
        shutil.copy(file_path, destination_path)
        print(f"Copied {file_path} to {destination_path}")


def _remove_stl_files(source_directory: str | Path) -> None:
    for filename in os.listdir(source_directory):
        if filename.endswith(".stl"):
            file_path = os.path.join(source_directory, filename)
            os.remove(file_path)


class Robot:
    """A class to adapt the world in a Mujoco XML file."""

    def __init__(self, robot_name: str, output_dir: str | Path, compiler: Compiler | None = None) -> None:
        """Initialize the robot.

        Args:
            robot_name (str): The name of the robot.
            output_dir (str | Path): The output directory.
            compiler (Compiler, optional): The compiler settings. Defaults to None.
        """
        self.robot_name = robot_name
        self.output_dir = output_dir
        self.compiler = compiler
        self._set_clean_up()
        self.tree = ET.parse(self.output_dir / f"{self.robot_name}.xml")

    def _set_clean_up(self) -> None:
        # HACK
        # mujoco has a hard time reading meshes
        _copy_stl_files(self.output_dir / "meshes", self.output_dir)
        # remove inertia tags
        urdf_tree = ET.parse(self.output_dir / f"{self.robot_name}.urdf")
        root = urdf_tree.getroot()
        for link in root.findall(".//link"):
            inertial = link.find("inertial")
            if inertial is not None:
                link.remove(inertial)

        tree = ET.ElementTree(root)
        tree.write(self.output_dir / f"{self.robot_name}.urdf", encoding="utf-8", xml_declaration=True)
        model = mujoco.MjModel.from_xml_path((self.output_dir / f"{self.robot_name}.urdf").as_posix())
        mujoco.mj_saveLastXML((self.output_dir / f"{self.robot_name}.xml").as_posix(), model)
        # remove all the files
        _remove_stl_files(self.output_dir)

    def adapt_world(self) -> None:
        root = self.tree.getroot()

        root.append(Option(
            timestep=0.001, viscosity=1e-6, iteration=50, solver="PGS", gravity=(0, 0, -9.81)
        ).to_xml())

        # TODO - check that
        root.append(
            Default(
                joint=Joint(limited=True),
                motor=Motor(ctrllimited=True),
                # TODO
                #geom=
                equality=Equality(solref=(0.001, 2))
                # visualgem?
                # joint param damping
            ).to_xml()
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

        new_root_body = Body(name="root", pos=(0, 0, 0), quat=(1, 0, 0, 0)).to_xml()
        # Move gathered elements to the new root body
        for item in items_to_move:
            worldbody.remove(item)
            new_root_body.append(item)

        # Add the new root body to the worldbody
        worldbody.append(new_root_body)

        worldbody.insert(
            0, 
            Light(
                directional=True, diffuse=(0.4, 0.4, 0.4), specular=(0.1, 0.1, 0.1), 
                pos=(0, 0, 5.0), dir=(0, 0, -1), castshadow=False
            ).to_xml()
        )
        worldbody.insert(
            0, 
            Light(
                directional=True, diffuse=(0.6, 0.6, 0.6), specular=(0.2, 0.2, 0.2), 
                pos=(0, 0, 4), dir=(0, 0, -1)
            ).to_xml()
        )
        worldbody.insert(
            0,
            Geom(mesh="ground", type="plane", pos=(0.001, 0, 0), quat=(1, 0, 0, 0),
                material="matplane", condim=1, conaffinity=15
            ).to_xml()
        )
    
        worldbody2 = worldbody
        # # Create a new root worldbody and add the new root body to it
        # new_worldbody = ET.Element("worldbody", name="new_root")
        index = list(root).index(worldbody)

        # # Remove the old <worldbody> and insert the new one at the same index
        root.remove(worldbody)
        # pfb30
        # root.insert(index, worldbody2)
        root.append(worldbody2)
        # modified_xml = ET.tostring(root, encoding="unicode")

        motors = []
        sensors = []
        for joint, limits in StompyFixed.default_limits().items():
            if joint in StompyFixed.legs.all_joints():
                motors.append(
                    Motor(
                        name=joint, joint=joint, gear=1, 
                        ctrlrange=(limits["lower"], limits["upper"]), ctrllimited=True
                    )
                )
                sensors.extend([
                        Actuatorpos(name=joint, actuator=joint, user="true"),
                        Actuatorvel(name=joint, actuator=joint, user="true"),
                        Actuatorfrc(name=joint, actuator=joint, user="true", noise=0.001)
                    ]
                )

        # # Add motors and sensors
        root.append(Actuator(motors).to_xml())
        root.append(Sensor(sensors).to_xml())


        # TODO pfb30 add visual
        # TODO add imu to the base

        self.tree = ET.ElementTree(root)
        rough_string = ET.tostring(self.tree.getroot(), 'utf-8')

        path = f"/Users/pfb30/sim-integration/13052024/robot_fixed.xml"
        import xml.dom.minidom
        def pretty_print_xml(xml_string):
            """Formats the provided XML string into a pretty-printed version."""
            parsed_xml = xml.dom.minidom.parseString(xml_string)
            return parsed_xml.toprettyxml(indent="  ")
        # Pretty print the XML
        formatted_xml = pretty_print_xml(rough_string)
                # Save the formatted XML to a file
        with open(path, 'w') as f:
            f.write(formatted_xml)
        # Pretty print the XML
        formatted_xml = pretty_print_xml(rough_string)
                # Save the formatted XML to a file
        with open(path, 'w') as f:
            f.write(formatted_xml)

    def save(self, path: str | Path) -> None:
        self.tree.write(path, encoding="utf-8") #, xml_declaration=True)


if __name__ == "__main__":
    robot_name = "robot_fixed"
    robot = Robot(
        robot_name, Path("/Users/pfb30/sim-integration/13052024"), 
        # pfb30 test eulerseq
        Compiler(angle="radian", meshdir="meshes", eulerseq="xyz"))
    robot.adapt_world()
    # robot.save(f"/Users/pfb30/sim-integration/13052024/{robot_name}.xml")




    # dom = xml.dom.minidom.parse(f"/Users/pfb30/sim-integration/13052024/{robot_name}.xml") # or xml.dom.minidom.parseString(xml_string)
    # pretty_xml_as_string = dom.toprettyxml()
