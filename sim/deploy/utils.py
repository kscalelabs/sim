"""Utility functions for the policy integration module."""

import numpy as np


def quaternion_to_euler_array(quat: np.ndarray) -> np.ndarray:
    """
    Converts a quaternion to Euler angles (roll, pitch, yaw).

    Args:
        quat: Quaternion in the format [x, y, z, w].

    Returns:
        Euler angles (roll, pitch, yaw) in radians.
    """
    # Ensure quaternion is in the correct format [x, y, z, w]
    x, y, z, w = quat

    # Roll (x-axis rotation)
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = np.arctan2(t0, t1)

    # Pitch (y-axis rotation)
    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch_y = np.arcsin(t2)

    # Yaw (z-axis rotation)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)

    # Returns roll, pitch, yaw in a NumPy array in radians
    return np.array([roll_x, pitch_y, yaw_z])
