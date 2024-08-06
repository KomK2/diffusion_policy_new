import zarr
import numpy as np

def downsample_data(data, factor):
    return data[::factor]

def downsample_episodes(data, episode_ends, factor):
    downsampled_data = []
    new_end_index_data = []
    start_index = 0

    for end_index in episode_ends:
        episode_data = data[start_index:end_index]
        downsampled_episode_data = downsample_data(episode_data, factor)
        new_end_index = len(downsampled_episode_data) + (new_end_index_data[-1] if new_end_index_data else 0)
        downsampled_data.append(downsampled_episode_data)
        new_end_index_data.append(new_end_index)
        start_index = end_index

    return np.concatenate(downsampled_data), np.array(new_end_index_data)

z = zarr.open(f'/home/bmv/diffusion_policy_new/data/real_crab_85/replay_buffer.zarr', mode='r')
episode_ends = z['meta/episode_ends'][:]

factor = 2  


# Downsample ft_data and episode_ends
# ft_data, downsampled_episode_ends = downsample_episodes(z['data/robot_eef_pose'][:], episode_ends, factor)

# Save the downsampled data to a new Zarr file
output_zarr_path = '/home/bmv/diffusion_policy_new/data/real_crab_85_20hz_down_samp/replay_buffer.zarr'
output_z = zarr.open(output_zarr_path, mode='w')
output_group = output_z.create_group('data')
meta_group = output_z.create_group('meta')


for name, item in z['data'].items():
    data = item[:]

    downsampled_data, downsample_end_index = downsample_episodes(data, episode_ends, factor)  # Correct factor to downsample from 50Hz to 10Hz
    
    # print(f'{name} after downsampling:', downsampled_data.shape)
    output_group.create_dataset(name, data=downsampled_data, shape=downsampled_data.shape, dtype=downsampled_data.dtype)

    print("end index:", downsample_end_index)

# Save the downsampled ft_data
# output_group.create_dataset('ft_data', data=ft_data, shape=ft_data.shape, chunks=True, dtype=ft_data.dtype)

# Save the new episode ends
meta_group.create_dataset('episode_ends', data=downsample_end_index, shape=downsample_end_index.shape, dtype=downsample_end_index.dtype)

# Copy other metadata items from the original file
for name, item in z['meta'].items():
    if name != 'episode_ends':
        data = item[:]
        meta_group.create_dataset(name, data=data, shape=data.shape, chunks=True, dtype=data.dtype)

print("Downsampling complete")

