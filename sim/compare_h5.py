"""
Compare two HDF5 files and analyze their differences.

Usage:
    python compare_h5.py --isaac-file path/to/isaac.h5 --mujoco-file path/to/mujoco.h5

Example:
    python sim/compare_h5.py \
        --isaac-file runs/h5_out/stompypro/2024-12-02_20-04-51/env_0/stompypro_env_0/h5_out/2024-12-02_20-04-51__053b2b5b-21c9-497c-b637-b66935dfe475.h5 \
        --mujoco-file sim/resources/stompypro/h5_out/2024-12-02_21-10-41__5820ade7-9fc0-46df-8469-2f305480bcae.h5
        
    python sim/compare_h5.py \
        --isaac-file runs/h5_out/stompypro/2024-12-02_20-04-51/env_1/stompypro_env_1/h5_out/2024-12-02_20-04-51__d2016a6f-a486-4e86-8c5e-c9addd2cc13e.h5 \
        --mujoco-file sim/resources/stompypro/h5_out/2024-12-02_21-10-41__5820ade7-9fc0-46df-8469-2f305480bcae.h5
        
"""

import argparse
from pathlib import Path

import h5py
import numpy as np


def load_h5_file(file_path):
    """Load H5 file and return a dictionary of datasets"""
    data = {}
    with h5py.File(file_path, "r") as f:
        # Recursively load all datasets
        def load_group(group, prefix=""):
            for key in group.keys():
                path = f"{prefix}/{key}" if prefix else key
                if isinstance(group[key], h5py.Group):
                    load_group(group[key], path)
                else:
                    data[path] = group[key][:]

        load_group(f)
    return data


def compare_h5_files(issac_path, mujoco_path):
    """Compare two H5 files and print differences"""
    print(f"\nLoading files:")
    print(f"Isaac: {issac_path}")
    print(f"Mujoco: {mujoco_path}")

    # Load both files
    issac_data = load_h5_file(issac_path)
    mujoco_data = load_h5_file(mujoco_path)

    print("\nFile lengths:")
    print(f"Isaac datasets: {len(issac_data)}")
    print(f"Mujoco datasets: {len(mujoco_data)}")

    print("\nDataset shapes:")
    print("\nIsaac shapes:")
    for key, value in issac_data.items():
        print(f"{key}: {value.shape}")

    print("\nMujoco shapes:")
    for key, value in mujoco_data.items():
        print(f"{key}: {value.shape}")

    # Find common keys
    common_keys = set(issac_data.keys()) & set(mujoco_data.keys())
    print(f"\nCommon datasets: {len(common_keys)}")

    # Find uncommon keys
    issac_only_keys = set(issac_data.keys()) - common_keys
    mujoco_only_keys = set(mujoco_data.keys()) - common_keys
    print(f"\nIsaac only datasets: {len(issac_only_keys)}")
    for key in issac_only_keys:
        print(f"Isaac only: {key}")
    print(f"\nMujoco only datasets: {len(mujoco_only_keys)}")
    for key in mujoco_only_keys:
        print(f"Mujoco only: {key}")

    # Compare data for common keys
    for key in common_keys:
        issac_arr = issac_data[key]
        mujoco_arr = mujoco_data[key]

        print(f"\n========== For {key} ===============")

        if issac_arr.shape != mujoco_arr.shape:
            print(f"\n{key} - Shape mismatch: Isaac {issac_arr.shape} vs Mujoco {mujoco_arr.shape}")

        # Calculate differences
        min_shape = min(issac_arr.shape[0], mujoco_arr.shape[0])
        if issac_arr.shape != mujoco_arr.shape:
            raise ValueError(f"Shapes do not match for {key}. Cannot compare datasets with different shapes.")
        diff = np.abs(issac_arr[:min_shape] - mujoco_arr[:min_shape])
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)

        print(f"Max difference: {max_diff:.6f}")
        print(f"Mean difference: {mean_diff:.6f}\n")

        start_idx = 0
        display_timesteps = 10
        end_idx = start_idx + display_timesteps

        np.set_printoptions(formatter={"float": "{:0.6f}".format}, suppress=True)
        print("Isaac:\n", issac_arr[start_idx:end_idx])
        print("Mujoco:\n", mujoco_arr[start_idx:end_idx])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare two H5 files from Isaac and Mujoco simulations")
    parser.add_argument("--isaac-file", required=True, help="Path to Isaac simulation H5 file")
    parser.add_argument("--mujoco-file", required=True, help="Path to Mujoco simulation H5 file")

    args = parser.parse_args()

    print(f"Isaac path: {args.isaac_file}")
    print(f"Mujoco path: {args.mujoco_file}")

    compare_h5_files(args.isaac_file, args.mujoco_file)
