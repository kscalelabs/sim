"""
1. We start from random q - which are actions!
2. We compute forward dynamics with pinnochio and get poses in general frame
3. The casadi takes this poses and computes against motion cap poses for keypoints
4. The objective is to minimize the difference between robot keypoints and MoCap keypoints
5. How do we get dynamics ?? We can get dynamics from pinnochio??
6. We iterate with casadi where obj function is given by pinnochio



"""
import numpy as np
import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper
import casadi as ca

# Load the MoCap data
mocap_data = np.load('mocap_data.npy')  # Assuming the data is stored in this file

# Define the robot model
robot = RobotWrapper.BuildFromURDF('/path/to/your/robot.urdf', ["/path/to/your/robot/meshes"])
model = robot.model
data = robot.data

# Number of frames and keypoints
N = mocap_data.shape[0]
n_keypoints = 10

# Define optimization variables
q = ca.MX.sym('q', model.nq)
qd = ca.MX.sym('qd', model.nv)

# Function to compute the positions of the keypoints on the robot
def robot_keypoints(q):
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    keypoints = []
    for i in range(n_keypoints):
        frame_id = model.getFrameId(f'keypoint_{i}')  # Assuming keypoints are named 'keypoint_0', 'keypoint_1', etc.
        keypoints.append(data.oMf[frame_id].translation)
    return ca.vertcat(*keypoints)

# Define the objective function and constraints
objective = 0
constraints = []
dt = 0.01  # Time step
q_traj = [ca.MX.sym(f'q_{t}', model.nq) for t in range(N)]
qd_traj = [ca.MX.sym(f'qd_{t}', model.nv) for t in range(N)]

for t in range(N):
    q_t = q_traj[t]
    mocap_t = mocap_data[t].reshape(-1)
    robot_kp_t = robot_keypoints(q_t).reshape(-1)
    
    # Objective: minimize the difference between robot keypoints and MoCap keypoints
    objective += ca.sumsqr(robot_kp_t - mocap_t)
    
    if t > 0:
        q_prev = q_traj[t-1]
        qd_prev = qd_traj[t-1]
        qd_t = qd_traj[t]
        
        # Constraint: q[t+1] = q[t] + (qd[t] + qd[t-1]) / 2 * dt
        constraints.append(q_t - (q_prev + (qd_t + qd_prev) / 2 * dt))

# Create the NLP solver
nlp = {
    'x': ca.vertcat(*q_traj, *qd_traj),
    'f': objective,
    'g': ca.vertcat(*constraints)
}

opts = {
    'ipopt.print_level': 0,
    'ipopt.max_iter': 1000,
    'ipopt.tol': 1e-6
}

solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

# Solve the optimization problem
sol = solver(x0=np.zeros((model.nq + model.nv) * N), lbg=0, ubg=0)

# Extract the solution
q_sol = ca.reshape(sol['x'][:model.nq * N], (N, model.nq))
qd_sol = ca.reshape(sol['x'][model.nq * N:], (N, model.nv))

# Print the results
print('Optimized joint positions (q):', q_sol)
print('Optimized joint velocities (qd):', qd_sol)