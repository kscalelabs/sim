"""Parses the URDF file and prints the joint names and types.
Assumes name-based heirarchy of URDF joints, with underscores as delimiters.
Additionally, will print out associated limits of revolute joints from the URDF."""

import argparse
import xml.etree.ElementTree as ET
from typing import Dict, List


def main() -> None:
    parser = argparse.ArgumentParser(description="Gets the links and joints for a URDF")
    parser.add_argument("urdf_path", help="The path to the URDF file")
    parser.add_argument("--ignore-joint-type", nargs="*", default=["fixed"], help="The joint types to ignore")
    args = parser.parse_args()

    ignore_joint_type = set(args.ignore_joint_type)

    with open(args.urdf_path, "r") as urdf_file:
        urdf = ET.parse(urdf_file).getroot()

    # Gets the relevant joint names.
    joint_names: List[str] = []
    joint_uppers: List[str] = []
    joint_lowers: List[str] = []
    for joint in urdf.findall("joint"):
        joint_type = joint.attrib["type"]
        if joint_type in ignore_joint_type:
            continue # ignoring fixed joints, or whatever is defined in arguments
        joint_name = joint.attrib["name"]
        joint_names.append(joint_name)
        joint_upper = joint.find('limit').get('upper')
        joint_lower = joint.find('limit').get('lower')
        joint_uppers.append(joint_upper)
        joint_lowers.append(joint_lower)
    
    
    for i in range(len(joint_names)):
        # Concatenate the limits into the name string
        joint_names[i] = joint_names[i] + " | Limits = [ Lower: " + joint_lowers[i] + ", Upper: " + joint_uppers[i] + " ]"
            # This is pretty bad way of doing this, but trying make a proper data storage system work with the below implementation
            #   of the tree structure became overly hard. The dict itself can store more data theoretically, but there's no clear way to 
            #   associated more data with the names as they're passed along and sorted into the tree.

# Makes a "tree" of the joints using common prefixes.
    joint_names.sort()
    joint_tree: Dict = {}
    for joint_name in joint_names:
        parts = joint_name.split("_")
        current_tree = joint_tree
        for part in parts:
            current_tree = current_tree.setdefault(part, {})

    # Collapses nodes with just one child.
    def collapse_tree(tree: Dict) -> None:
        for key, value in list(tree.items()):
            collapse_tree(value)
            if len(value) == 1:
                child_key, child_value = next(iter(value.items()))
                tree[key + "_" + child_key] = child_value
                del tree[key]

    collapse_tree(joint_tree)

    # Replaces leaf nodes with their full names.
    def replace_leaves(tree: Dict, prefix: str) -> None:
        for key, value in list(tree.items()):
            if not value:
                tree[key] = prefix + key
            else:
                replace_leaves(value, prefix + key + "_")

    replace_leaves(joint_tree, "")

    # Prints the tree.
    def print_tree(tree: Dict, depth: int = 0) -> None:
        for key, value in tree.items():
            if isinstance(value, dict):
                print("  " * depth + key)
                print_tree(value, depth + 1)
            else:
                print("  " * depth + key) #+ ": " + value)

    print_tree(joint_tree)


if __name__ == "__main__":
    # python -m sim.scripts.print_joints
    main()
