import zarr
import numpy as np

# Load the Zarr dataset
z = zarr.open('/home/bmv/diffusion_policy_new/data/replicaless_rice_scoop/replay_buffer.zarr', mode='a')

poses = z['data/replica_eef_pose'][:]
ep_len = z['meta/episode_ends'][:]

# Initialize a list to store deltas
deltas = []

# Loop through each episode to calculate deltas
for episode_index in range(len(ep_len)):
    start_index = 0 if episode_index == 0 else ep_len[episode_index - 1]
    end_index = ep_len[episode_index]

    x1 = poses[start_index:end_index, 0]
    y1 = poses[start_index:end_index, 1]
    z1 = poses[start_index:end_index, 2]
    rx1 = poses[start_index:end_index, 3]
    ry1 = poses[start_index:end_index, 4]
    rz1 = poses[start_index:end_index, 5]

    if episode_index == 0:
        # For the first episode, append zeros since no previous data exists
        deltas.append(np.zeros((end_index - start_index, 6)))
    else:
        # Get previous episode's last pose as the starting reference
        x0 = poses[start_index - 1:end_index - 1, 0]
        y0 = poses[start_index - 1:end_index - 1, 1]
        z0 = poses[start_index - 1:end_index - 1, 2]
        rx0 = poses[start_index - 1:end_index - 1, 3]
        ry0 = poses[start_index - 1:end_index - 1, 4]
        rz0 = poses[start_index - 1:end_index - 1, 5]

        # Calculate deltas
        deltax = x1 - x0
        deltay = y1 - y0
        deltaz = z1 - z0
        deltarx = rx1 - rx0
        deltary = ry1 - ry0
        deltarz = rz1 - rz0

        # Stack deltas to create a [num_samples, 6] array
        delta = np.stack([deltax, deltay, deltaz, deltarx, deltary, deltarz], axis=1)
        deltas.append(delta)

# Concatenate all deltas into a single array
deltas = np.concatenate(deltas, axis=0)

# Write the deltas into the Zarr dataset
z.create_dataset('data/replica_eef_deltas', data=deltas, overwrite=True)

print(f"Delta dataset written successfully with shape: {deltas.shape}")
