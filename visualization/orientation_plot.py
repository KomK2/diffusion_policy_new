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

    rotation_data = replica_eef_poses[:, 3:]

    


    # print("the rotation data is ", rotation_data)
    # Create a 3D plot
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')

    # # Set the limits (adjust based on your data)
    # ax.set_xlim([-5, 5])
    # ax.set_ylim([-5, 5])
    # ax.set_zlim([0, 2])

    # # Labels
    # ax.set_xlabel('Rx')
    # ax.set_ylabel('Ry')
    # ax.set_zlabel('Rz')

    # # Origin point for the arrows (you can use the actual (x, y, z) if you want to combine position and rotation)
    # for i in range(1, rotation_data.shape[0]):
    #     # Start from the previous point
    #     start = rotation_data[i - 1, :]
    #     # Calculate the difference (arrow direction)
    #     direction = rotation_data[i, :] - start
    #     # Plot the arrow

    #     print("the difference is ",direction )

    #     ax.quiver(
    #         start[0], start[1], start[2], 
    #         direction[0], direction[1], direction[2], 
    #         length=1, color='blue'
    #     )
    #     plt.pause(0.1)  # Pause to create an animation effect

    # plt.show()

