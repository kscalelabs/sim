"""Logger for logging data to HDF5 files."""

import os
import uuid
from datetime import datetime
from typing import Dict, Tuple

import h5py
import matplotlib.pyplot as plt  # dependency issues with python 3.8
import numpy as np


class HDF5Logger:
    def __init__(
        self,
        data_name: str,
        num_actions: int,
        max_timesteps: int,
        num_observations: int,
        h5_out_dir: str = "sim/resources/",
    ) -> None:
        self.data_name = data_name
        self.num_actions = num_actions
        self.max_timesteps = max_timesteps
        self.num_observations = num_observations
        self.max_threshold = 1e3  # Adjust this threshold as needed
        self.h5_out_dir = h5_out_dir
        self.h5_file, self.h5_dict = self._create_h5_file()
        self.current_timestep = 0

    def _create_h5_file(self) -> Tuple[h5py.File, Dict[str, h5py.Dataset]]:
        # Create a unique file ID
        idd = str(uuid.uuid4())
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        curr_h5_out_dir = f"{self.h5_out_dir}/{self.data_name}/h5_out/"
        os.makedirs(curr_h5_out_dir, exist_ok=True)

        h5_file_path = f"{curr_h5_out_dir}/{timestamp}__{idd}.h5"
        print(f"Saving HDF5 data to {h5_file_path}")
        h5_file = h5py.File(h5_file_path, "w")

        # Create datasets for logging actions and observations
        dset_prev_actions = h5_file.create_dataset(
            "prev_actions", (self.max_timesteps, self.num_actions), dtype=np.float32
        )
        dset_2d_command = h5_file.create_dataset("observations/2D_command", (self.max_timesteps, 2), dtype=np.float32)
        dset_3d_command = h5_file.create_dataset("observations/3D_command", (self.max_timesteps, 3), dtype=np.float32)
        dset_q = h5_file.create_dataset("observations/q", (self.max_timesteps, self.num_actions), dtype=np.float32)
        dset_dq = h5_file.create_dataset("observations/dq", (self.max_timesteps, self.num_actions), dtype=np.float32)
        dset_ang_vel = h5_file.create_dataset("observations/ang_vel", (self.max_timesteps, 3), dtype=np.float32)
        dset_euler = h5_file.create_dataset("observations/euler", (self.max_timesteps, 3), dtype=np.float32)
        dset_t = h5_file.create_dataset("observations/t", (self.max_timesteps, 1), dtype=np.float32)
        dset_buffer = h5_file.create_dataset(
            "observations/buffer", (self.max_timesteps, self.num_observations), dtype=np.float32
        )
        dset_curr_actions = h5_file.create_dataset(
            "curr_actions", (self.max_timesteps, self.num_actions), dtype=np.float32
        )

        # Map datasets for easy access
        h5_dict = {
            "prev_actions": dset_prev_actions,
            "curr_actions": dset_curr_actions,
            "2D_command": dset_2d_command,
            "3D_command": dset_3d_command,
            "joint_pos": dset_q,
            "joint_vel": dset_dq,
            "ang_vel": dset_ang_vel,
            "euler_rotation": dset_euler,
            "t": dset_t,
            "buffer": dset_buffer,
        }

        metadata = {
            "data_name": self.data_name,
            "num_actions": self.num_actions,
            "num_observations": self.num_observations,
            "max_timesteps": self.max_timesteps,
            "creation_time": timestamp,
        }
        h5_file.attrs['metadata'] = metadata

        return h5_file, h5_dict

    def log_data(self, data: Dict[str, np.ndarray]) -> None:
        if self.current_timestep >= self.max_timesteps:
            print(f"Warning: Exceeded maximum timesteps ({self.max_timesteps})")
            return

        for key, dataset in self.h5_dict.items():
            if key in data:
                if data[key].shape != dataset.shape[1:]:
                    print(f"Warning: Data shape mismatch for {key}. Expected {dataset.shape[1:]}, got {data[key].shape}.")
                    continue
                dataset[self.current_timestep] = data[key]

        self.current_timestep += 1

    def close(self) -> None:
        for key, dataset in self.h5_dict.items():
            max_val = np.max(np.abs(dataset[:]))
            if max_val > self.max_threshold:
                print(f"Warning: Found very large values in {key}: {max_val}")
                print("File will not be saved to prevent corrupted data")
                self.h5_file.close()
                # Delete the file
                os.remove(self.h5_file.filename)
                return

        self.h5_file.close()

    @staticmethod
    def visualize_h5(h5_file_path: str, variable: str = None) -> None:
        """Visualizes the data from an HDF5 file by plotting each variable one by one.

        Args:
            h5_file_path (str): Path to the HDF5 file.
            variable (str, optional): Specific variable to visualize. If None, all variables are plotted.
        """
        try:
            with h5py.File(h5_file_path, "r") as h5_file:
                keys = [variable] if variable else h5_file.keys()
                for key in keys:
                    if key in h5_file:
                        group = h5_file[key]
                        if isinstance(group, h5py.Group):
                            for subkey in group.keys():
                                dataset = group[subkey][:]
                                HDF5Logger._plot_dataset(f"{key}/{subkey}", dataset)
                        else:
                            dataset = group[:]
                            HDF5Logger._plot_dataset(key, dataset)

        except Exception as e:
            print(f"Failed to visualize HDF5 file: {e}")

    @staticmethod
    def _plot_dataset(name: str, data: np.ndarray) -> None:
        """Helper method to plot a single dataset.

        Args:
            name (str): Name of the dataset.
            data (np.ndarray): Data to be plotted.
        """
        plt.figure(figsize=(10, 5))
        if data.ndim == 2:  # Handle multi-dimensional data
            for i in range(data.shape[1]):
                plt.plot(data[:, i], label=f"{name}[{i}]")
        else:
            plt.plot(data, label=name)

        plt.title(f"Visualization of {name}")
        plt.xlabel("Timesteps")
        plt.ylabel("Values")
        plt.legend(loc="upper right")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
