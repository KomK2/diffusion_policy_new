
import zarr

# Load the Zarr file
zarr_data = zarr.open('/home/bmv/diffusion_policy_new/data/ft_100hz/replay_buffer.zarr', mode='r')

# List the groups and datasets within the Zarr file
print(zarr_data.tree())

# Access a specific dataset, for example 'data/action/0.0'
action_data = zarr_data['data/action/0.0'][:]
print("Action data sample:", action_data)

# You can also explore other datasets like 'ft_data' or 'replica_eef_pose'
ft_data = zarr_data['data/ft_data/0.0.0'][:]
print("Force-torque data sample:", ft_data)

replica_eef_pose_data = zarr_data['data/replica_eef_pose/0.0'][:]
print("Replica EEF pose data sample:", replica_eef_pose_data)

# Access episode ends metadata
episode_ends = zarr_data['meta/episode_ends/0'][:]
print("Episode ends metadata:", episode_ends)
