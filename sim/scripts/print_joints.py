"""Parses the URDF file and prints the joint names and types."""

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
    for joint in urdf.findall("joint"):
        joint_type = joint.attrib["type"]
        if joint_type in ignore_joint_type:
            continue
        joint_name = joint.attrib["name"]
        joint_names.append(joint_name)

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
                print("  " * depth + key + ": " + value)

    print_tree(joint_tree)


if __name__ == "__main__":
    # python -m sim.scripts.print_joints
    main()
