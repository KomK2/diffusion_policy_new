import zarr
import numpy as np
from visual_kinematics.RobotSerial import *
from visual_kinematics.RobotTrajectory import *
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import Circle
from math import pi
import scipy.spatial.transform as st
from viz_constants import VIZ_DIR

dataset_dir = VIZ_DIR
# Load the Zarr dataset
z = zarr.open(f'/home/bmv/diffusion_policy_new/data/real_crab_85_half_down_samp/replay_buffer.zarr', mode='r')

poses = z['data/replica_eef_pose'][:]
# qpos = z['data/robot_joint'][:]
ep_len = z['meta/episode_ends'][:]
ts = z['data/timestamp'][:]

# Determine start and end index of the episode
print(f"Visualizing Dataset: {dataset_dir}")
episode_index = int(input("Enter episode index to visualize || -1 for all data || -2 for 5 consecutive episodes: "))

# Visualize all trajectories
if episode_index == -1:
    # Extract x, y, z coordinates and convert to millimeters
    x = poses[:, 0] * 1000
    y = poses[:, 1] * 1000
    z = poses[:, 2] * 1000

# Visualize 5 consecutive trajectories
elif episode_index == -2:
    # Extract x, y, z coordinates for 5 consecutive episodes and convert to millimeters
    episode_start = 0
    colors = ['b', 'g', 'r', 'c', 'm']
    labels = [f'Episode {i}' for i in range(episode_start + 1, episode_start + 6)]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i in range(5):
        ep_idx = episode_start + i
        start_index = 0 if ep_idx == 0 else ep_len[ep_idx - 1]
        end_index = ep_len[ep_idx]
        x = poses[:, 0][start_index:end_index] * 1000
        y = poses[:, 1][start_index:end_index] * 1000
        z = poses[:, 2][start_index:end_index] * 1000
        ax.scatter(x, y, z, c=colors[i], label=labels[i])

# Visualize specific trajectory
else:
    start_index = 0 if episode_index == 0 else ep_len[episode_index - 1]
    end_index = ep_len[episode_index]

    x = poses[:, 0][start_index:end_index] * 1000
    y = poses[:, 1][start_index:end_index] * 1000
    z = poses[:, 2][start_index:end_index] * 1000

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c='b', label=f'Episode {episode_index}')
    print("Poses in the episode are: ", poses[start_index:end_index])

# Set labels and title
ax.set_xlabel('X (mm)')
ax.set_ylabel('Y (mm)')
ax.set_zlabel('Z (mm)')
ax.set_title('Robot Cartesian Position')

# Adjust the limits for the axes (converted to millimeters)
ax.set_xlim([-100, 100])  # Set limits for the x-axis in mm
ax.set_ylim([-400, -200])  # Set limits for the y-axis in mm
ax.set_zlim([100, 300])  # Set limits for the z-axis in mm

# Add legend
ax.legend()

plt.show()
