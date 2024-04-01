"""DOF control methods example
---------------------------
An example that demonstrates various DOF control methods:
- Load cartpole asset from an urdf
- Get/set DOF properties
- Set DOF position and velocity targets
- Get DOF positions
- Apply DOF efforts

Copyright KScale Labs.
"""

import torch
from isaacgym import gymapi, gymutil

# initialize gym
gym = gymapi.acquire_gym()

# parse arguments
args = gymutil.parse_arguments(description="Joint control Methods Example")

# create a simulator
sim_params = gymapi.SimParams()
sim_params.substeps = 2
sim_params.dt = 1.0 / 60.0

sim_params.physx.solver_type = 1
sim_params.physx.num_position_iterations = 4
sim_params.physx.num_velocity_iterations = 1

sim_params.physx.num_threads = args.num_threads
sim_params.physx.use_gpu = args.use_gpu

sim_params.use_gpu_pipeline = False
if args.use_gpu_pipeline:
    print("WARNING: Forcing CPU pipeline.")

# pfb30
# sim_params.gravity = gymapi.Vec3(0.0, -10.0, 0.0)

sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)

if sim is None:
    print("*** Failed to create sim")
    quit()

# create viewer using the default camera properties
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    raise ValueError("*** Failed to create viewer")

# add ground plane
plane_params = gymapi.PlaneParams()
gym.add_ground(sim, gymapi.PlaneParams())

# set up the env grid
num_envs = 1
spacing = 1.5
env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
env_upper = gymapi.Vec3(spacing, 0.0, spacing)


# add cartpole urdf asset
asset_root = "../sim-integration/humanoid-gym/resources/robots/XBot/"
asset_file = "urdf_whole_body/XBot-L_example.urdf"
# asset_file = "urdf/XBot-L.urdf"

asset_root = "../sim-integration/humanoid-gym/resources/robot_new4/"
asset_file = "legs.urdf"

asset_root = "../sim-integration/humanoid-gym/resources/test_onshape/"
asset_file = "test.urdf"
print(asset_file)

# Load asset with default control type of position for all joints
asset_options = gymapi.AssetOptions()
# pfb30 - set mode pose
asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
print("Loading asset '%s' from '%s'" % (asset_file, asset_root))
robot_asset = gym.load_asset(sim, asset_root, asset_file, asset_options)


# initial root pose for cartpole actors
initial_pose = gymapi.Transform()
initial_pose.p = gymapi.Vec3(0.0, 9.8, 0.0)
# initial_pose.r = gymapi.Quat(-0.717107, 0.0, 0.0, 0.717107)
# initial_pose.r = gymapi.Quat(0.7109, 0.7033, 0,0 )
# [0.0000, 0.7033, 0.0000, 0.7109]
# Create environment 1
env1 = gym.create_env(sim, env_lower, env_upper, 1)

robot1 = gym.create_actor(env1, robot_asset, initial_pose, "robot1")


# gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
body_props = gym.get_actor_rigid_body_properties(env1, robot1)

# print(f"First", body_props[0].mass)
# mass = 0
# for ii, body in enumerate(body_props):
#     body.mass = abs(body.mass)*0.01
#     mass += body.mass
#     print(ii, body.mass)
# print(f"First again", body_props[0].mass)
# print("total mass", mass)
# breakpoint()

# Configure DOF properties
props = gym.get_actor_dof_properties(env1, robot1)

# props["driveMode"] = (gymapi.DOF_MODE_EFFORT)
# # Required to work with dof effort
# props["damping"].fill(0.0)
# props["stiffness"].fill(0.0)

# for pos control stiffness high low damping
props["driveMode"] = gymapi.DOF_MODE_POS
props["stiffness"].fill(200.0)
props["damping"].fill(10.0)
props["armature"].fill(0.0001)
gym.set_actor_dof_properties(env1, robot1, props)


# # I. test the default position
# for ii in range(props.shape[0]):
#     gym.set_dof_target_position(env1, ii, 0)

# # II. test setting the position
left_knee_joint = gym.find_actor_dof_handle(env1, robot1, "left_knee_joint")
right_knee_joint = gym.find_actor_dof_handle(env1, robot1, "right_knee_joint")
right_shoulder_pitch_joint = gym.find_actor_dof_handle(env1, robot1, "right_shoulder_pitch_joint")
left_shoulder_pitch_joint = gym.find_actor_dof_handle(env1, robot1, "left_shoulder_pitch_joint")
right_elbow_pitch_joint = gym.find_actor_dof_handle(env1, robot1, "right_elbow_pitch_joint")
left_elbow_pitch_joint = gym.find_actor_dof_handle(env1, robot1, "left_elbow_pitch_joint")

# gym.set_dof_target_position(env1, left_knee_joint, 1.1)
# gym.set_dof_target_position(env1, right_knee_joint, -1.1)
# gym.set_dof_target_position(env1, right_shoulder_pitch_joint, -1.4)
# gym.set_dof_target_position(env1, left_shoulder_pitch_joint, 1.4)
# gym.set_dof_target_position(env1, right_elbow_pitch_joint, -2.)
# gym.set_dof_target_position(env1, left_elbow_pitch_joint, 2.)

# Look at the first env
cam_pos = gymapi.Vec3(8, 4, 1.5)
cam_target = gymapi.Vec3(0, 2, 1.5)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)


# Simulate
while not gym.query_viewer_has_closed(viewer):

    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # update the viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)

    # # Update env 1: make the robot always bend the knees
    # Set DOF drive targets
    # breakpoint()
    # torques = torch.zeros(props["damping"].shape)
    # gym.set_dof_actuation_force_tensor(env1, torques)

    # Apply dof effort
    # gym.apply_dof_effort(env1, right_knee_joint, 20)
    # gym.apply_dof_effort(env1, left_knee_joint, -20)

    # Set position

    # Position 1 elbows:
    states = torch.zeros(props.shape[0])

    # gym.set_dof_state_tensor(env1, states)

    # gym.set_actor_dof_position_targets(env1, right_elbow_pitch_joint, 2.1)
    # gym.set_actor_dof_position_targets(env1, left_elbow_pitch_joint, -2.1)
    # Get positions
    pos = gym.get_dof_position(env1, right_shoulder_pitch_joint)
    pos2 = gym.get_dof_position(env1, left_shoulder_pitch_joint)
    # print(pos, pos2)

    # Wait for dt to elapse in real time.
    # This synchronizes the physics simulation with the rendering rate.
    gym.sync_frame_time(sim)

print("Done")

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
