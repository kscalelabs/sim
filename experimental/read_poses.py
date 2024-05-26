import joblib
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import pinocchio as pin

# Load data
PATH = "data/demo_vid.pkl"
results = joblib.load(PATH)

JOINTS = 45
num_frames = len(results)
indices = [ii for ii in range(0, 45)]

# keypoints = {
#     "head": 18,  # good
#     "neck": 40,  # good
#     "shoulder_right": 2,
#     "shoulder_left": 34,
#     "torso": 27,
#     "elbow_right": 3,
#     "elbow_left": 6,
#     "wrist_right": 31,  # good
#     "wrist_left": 36, # good
#     "knee_right":  26,  # good
#     "knee_left":  29,  # good
#     "foot_right":  21, # good
#     "foot_left":  25,   # good
# }
# indices = [keypoints[key] for key in keypoints]

# indices = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
# Extract 3D joint data
three_d_joints = []
for key, item in results.items():
    three_d_joints.append(item["3d_joints"][0][indices])


# Rotation matrix for 180 degrees rotation around the X-axis
rotation_matrix_x = np.array([
    [1, 0, 0],
    [0, -1, 0],
    [0, 0, -1]
])

def rotate_joints(joints, rotation_matrix):
    return np.dot(joints, rotation_matrix)

def visualize_3d_joints(three_d_joints):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    def update_graph(num):
        ax.clear()
        ax.set_title(f'Frame {num + 1}')
        joints = three_d_joints[num]
        xs, ys, zs = joints[:, 0], joints[:, 1], joints[:, 2]
        ax.scatter(xs, ys, zs)

        # Add joint numbers
        for i, (x, y, z) in enumerate(zip(xs, ys, zs)):
            ax.text(x, y, z, str(i), color='red', fontsize=8)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])

    # Create an animation
    ani = FuncAnimation(fig, update_graph, frames=num_frames, interval=33)

    # Save the animation as a gif or display it
    ani.save('3d_joints_animation.gif', writer='imagemagick')
    # plt.show()

# Visualize the 3D joints
visualize_3d_joints(three_d_joints[:num_frames])