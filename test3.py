import zarr
import numpy as np

# Load the Zarr dataset
z = zarr.open('/home/bmv/diffusion_policy_new/data/rice_scoop_teleop_2/replay_buffer.zarr', mode='r')

replica_poses = z['data/replica_eef_pose'][:]
ep_len = z['meta/episode_ends'][:]

new_replica_poses = []
new_ep_len = []

start_index = 0
for episode_index in range(len(ep_len)):
    end_index = ep_len[episode_index]
    
    episode_data = replica_poses[start_index:end_index]
    
    filtered_data = episode_data[~np.all(episode_data == 0, axis=1)]
    
    new_replica_poses.append(filtered_data)
    
    if len(new_ep_len) == 0:
        new_ep_len.append(len(filtered_data))
    else:
        new_ep_len.append(new_ep_len[-1] + len(filtered_data))
    
    start_index = end_index

new_replica_poses = np.concatenate(new_replica_poses, axis=0)

new_ep_len = np.array(new_ep_len)

