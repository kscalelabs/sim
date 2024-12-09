import numpy as np


def debug_robot_state(robot_name: str, obs_buf: np.ndarray, actions: np.ndarray) -> None:
    """Debug function for robot state from observation buffer."""
    # Unpack the obs_buf components
    cmd = obs_buf[:5]  # Command input (sin, cos, vel_x, vel_y, vel_yaw)
    q = obs_buf[5:21]  # Joint positions (16)
    dq = obs_buf[21:37]  # Joint velocities (16)
    actions = obs_buf[37:53]  # Actions (16)
    ang_vel = obs_buf[53:56]  # Base angular velocity (3)
    euler = obs_buf[56:59]  # Base euler angles (3)

    quat = euler_to_quat(euler)

    # Detailed unpack
    vel_x = cmd[2]
    vel_y = cmd[3]
    vel_yaw = cmd[4]

    # Print the robot state
    print(f"\n=== {robot_name} State ===")
    print(f"Command: {vel_x=:.2f} {vel_y=:.2f} {vel_yaw=:.2f}")
    print(f"RPY: [{euler[0]:.2f}, {euler[1]:.2f}, {euler[2]:.2f}]")
    print(f"Quat: [{quat[0]:.2f}, {quat[1]:.2f}, {quat[2]:.2f}, {quat[3]:.2f}]")
    print(f"AngVel: [{ang_vel[0]:.2f}, {ang_vel[1]:.2f}, {ang_vel[2]:.2f}]")

    print("Joints:")
    print("  Pos:", " ".join(f"{x:5.2f}" for x in q))
    print("  Vel:", " ".join(f"{x:5.2f}" for x in dq))
    print("  Act:", " ".join(f"{x:5.2f}" for x in actions))
    
    print("Actions:")
    print("  ", " ".join(f"{x:5.2f}" for x in actions))
    print("=================")


def euler_to_quat(euler):
    """Convert euler angles (roll, pitch, yaw) to quaternion (x, y, z, w)."""
    roll, pitch, yaw = euler

    cr, cp, cy = np.cos(roll / 2), np.cos(pitch / 2), np.cos(yaw / 2)
    sr, sp, sy = np.sin(roll / 2), np.sin(pitch / 2), np.sin(yaw / 2)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    return np.array([qx, qy, qz, qw])
