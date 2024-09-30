import zarr
import numpy as np
from visual_kinematics.RobotSerial import *
from visual_kinematics.RobotTrajectory import *
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import Circle
from math import pi
import scipy.spatial.transform as st

z = zarr.open(f'/home/bmv/diffusion_policy_new/data/rice_scoop_both/replay_buffer.zarr', mode='r')

replica_eef_poses = z['data/replica_eef_pose'][:]
robot_eef_pose = z['data/robot_eef_pose'][:]
replica_joint = z['data/replica_joint'][:]
robot_joint = z['data/robot_joint'][:]
ep_len = z['meta/episode_ends'][:]
ts = z['data/timestamp'][:]

episode_index = int(input("Enter episode index to visualize || -1 for all data: "))

if episode_index != -1:
    start_index = 0 if episode_index == 0 else ep_len[episode_index - 1]
    end_index = ep_len[episode_index]

    replica_eef_poses = replica_eef_poses[start_index:end_index]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Set limits
    ax.set_xlim([-0.2, 0.2])
    ax.set_ylim([-0.2, 0.2])
    ax.set_zlim([0.2, 0.5])

    # Labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Plot each point one by one
    for i in range(replica_eef_poses.shape[0]):
        ax.scatter(replica_eef_poses[i, 0], replica_eef_poses[i, 1], replica_eef_poses[i, 2], color='blue')
        plt.pause(0.1)  # Pause to create an animation effect

    plt.show()



# if episode_index == -1:
#     # Extract x, y, z, rx, ry, rz coordinates for all episodes
#     x = poses[:, 0]
#     y = poses[:, 1]
#     z = poses[:, 2]
#     rx = poses[:, 3]
#     ry = poses[:, 4]
#     rz = poses[:, 5]

#     x1 = robot_pose[:, 0]
#     y1 = robot_pose[:, 1]
#     z1 = robot_pose[:, 2]
#     rx1 = robot_pose[:, 3]
#     ry1 = robot_pose[:, 4]
#     rz1 = robot_pose[:, 5]

#     j1 = qpos[:, 0]
#     j2 = qpos[:, 1]
#     j3 = qpos[:, 2]
#     j4 = qpos[:, 3]
#     j5 = qpos[:, 4]
#     j6 = qpos[:, 5]

#     rj1 = robot_joint[:, 0]
#     rj2 = robot_joint[:, 1]
#     rj3 = robot_joint[:, 2]
#     rj4 = robot_joint[:, 3]
#     rj5 = robot_joint[:, 4]
#     rj6 = robot_joint[:, 5]

# else:
#     # Extract specific episode
#     start_index = 0 if episode_index == 0 else ep_len[episode_index - 1]
#     end_index = ep_len[episode_index]

#     x = poses[start_index:end_index, 0]
#     y = poses[start_index:end_index, 1]
#     z = poses[start_index:end_index, 2]
#     rx = poses[start_index:end_index, 3]
#     ry = poses[start_index:end_index, 4]
#     rz = poses[start_index:end_index, 5]

#     x1 = robot_pose[start_index:end_index, 0]
#     y1 = robot_pose[start_index:end_index, 1]
#     z1 = robot_pose[start_index:end_index, 2]
#     rx1 = robot_pose[start_index:end_index, 3]
#     ry1 = robot_pose[start_index:end_index, 4]
#     rz1 = robot_pose[start_index:end_index, 5]

# # Compute absolute differences
# abs_diff_x = np.abs(x - x1)
# abs_diff_y = np.abs(y - y1)
# abs_diff_z = np.abs(z - z1)
# abs_diff_rx = np.abs(rx - rx1)
# abs_diff_ry = np.abs(ry - ry1)
# abs_diff_rz = np.abs(rz - rz1)

# joint_diff1 = np.abs(j1 - rj1)
# joint_diff2 = np.abs(j2 - rj2)
# joint_diff3 = np.abs(j3 - rj3)
# joint_diff4 = np.abs(j4 - rj4)
# joint_diff5 = np.abs(j5 - rj5)
# joint_diff6 = np.abs(j6 - rj6)

# # Compute mean and standard deviation of differences
# mean_diff_x = np.mean(abs_diff_x) * 1000
# mean_diff_y = np.mean(abs_diff_y) * 1000
# mean_diff_z = np.mean(abs_diff_z) * 1000
# mean_diff_rx = np.mean(abs_diff_rx) * (180 / np.pi)
# mean_diff_ry = np.mean(abs_diff_ry) * (180 / np.pi)
# mean_diff_rz = np.mean(abs_diff_rz) * (180 / np.pi)

# std_diff_x = np.std(abs_diff_x) * 1000
# std_diff_y = np.std(abs_diff_y) * 1000
# std_diff_z = np.std(abs_diff_z) * 1000
# std_diff_rx = np.std(abs_diff_rx) * (180 / np.pi)
# std_diff_ry = np.std(abs_diff_ry) * (180 / np.pi)
# std_diff_rz = np.std(abs_diff_rz) * (180 / np.pi)
# ##compute mean and standard deviation of joint differences
# mean_diff_j1 = np.mean(joint_diff1) * (180 / np.pi)
# mean_diff_j2 = np.mean(joint_diff2) * (180 / np.pi)
# mean_diff_j3 = np.mean(joint_diff3) * (180 / np.pi)
# mean_diff_j4 = np.mean(joint_diff4) * (180 / np.pi)
# mean_diff_j5 = np.mean(joint_diff5) * (180 / np.pi)
# mean_diff_j6 = np.mean(joint_diff6) * (180 / np.pi)

# std_diff_j1 = np.std(joint_diff1) * (180 / np.pi)
# std_diff_j2 = np.std(joint_diff2) * (180 / np.pi)
# std_diff_j3 = np.std(joint_diff3) * (180 / np.pi)
# std_diff_j4 = np.std(joint_diff4) * (180 / np.pi)
# std_diff_j5 = np.std(joint_diff5) * (180 / np.pi)
# std_diff_j6 = np.std(joint_diff6) * (180 / np.pi)
# # Print the results
# print("============================================")


# print(f"j1 Difference: Mean={mean_diff_j1:.4f}, Std Dev={std_diff_j1:.4f}")
# print(f"j2 Difference: Mean={mean_diff_j2:.4f}, Std Dev={std_diff_j2:.4f}")
# print(f"j3 Difference: Mean={mean_diff_j3:.4f}, Std Dev={std_diff_j3:.4f}")
# print(f"j4 Difference: Mean={mean_diff_j4:.4f}, Std Dev={std_diff_j4:.4f}")
# print(f"j5 Difference: Mean={mean_diff_j5:.4f}, Std Dev={std_diff_j5:.4f}")
# print(f"j6 Difference: Mean={mean_diff_j6:.4f}, Std Dev={std_diff_j6:.4f}")

# # Visualize the 3D scatter plot for the selected episode(s)
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(x, y, z, c='r', label='Replica Pose')
# ax.scatter(x1, y1, z1, c='b', label='Robot Pose')

# # Set labels and title
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# ax.set_title('Robot Poses')

# ax.set_xlim([-0.5, 0.5])
# ax.set_ylim([-1, 0])
# ax.set_zlim([0.0, 0.5])

# plt.legend()
# plt.show()
