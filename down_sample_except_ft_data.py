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

def downsample_ft_data(data, episode_ends, factor):
    downsampled_data = []
    new_end_index_data = []
    start_index = 0

    for end_index in episode_ends:
        episode_data = data[start_index:end_index]
        
        # Reshape the episode data to group into blocks of 'factor'
        num_blocks = len(episode_data) // factor
        reshaped_episode_data = episode_data[:num_blocks * factor].reshape(num_blocks, factor, -1)
        
        # If there are remaining elements that couldn't form a full block, pad with zeros
        if len(episode_data) % factor != 0:
            remaining_data = episode_data[num_blocks * factor:]
            padding = np.zeros((factor - remaining_data.shape[0], remaining_data.shape[1]))
            padded_remaining_data = np.vstack([remaining_data, padding])
            reshaped_episode_data = np.vstack([reshaped_episode_data, padded_remaining_data[None]])
        
        new_end_index = len(reshaped_episode_data) + (new_end_index_data[-1] if new_end_index_data else 0)
        downsampled_data.append(reshaped_episode_data)
        new_end_index_data.append(new_end_index)
        start_index = end_index

    return np.concatenate(downsampled_data), np.array(new_end_index_data)

# Load the Zarr file
z = zarr.open(f'/home/bmv/diffusion_policy_new/data/real_crab_85/replay_buffer.zarr', mode='r')
episode_ends = z['meta/episode_ends'][:]

factor = 5  

# Downsample ft_data and episode_ends
ft_data, downsampled_episode_ends = downsample_ft_data(z['data']['ft_data'][:], episode_ends, factor)

# Save the downsampled data to a new Zarr file
output_zarr_path = '/home/bmv/diffusion_policy_new/data/real_crab_85_10hz_down_samp/replay_buffer.zarr'
output_z = zarr.open(output_zarr_path, mode='w')
output_group = output_z.create_group('data')
meta_group = output_z.create_group('meta')

for name, item in z['data'].items():
    data = item[:]

    if name == 'ft_data':
        downsampled_data, downsample_end_index = downsample_ft_data(data, episode_ends, factor)
    else:
        downsampled_data, downsample_end_index = downsample_episodes(data, episode_ends, factor)  
    
    output_group.create_dataset(name, data=downsampled_data, shape=downsampled_data.shape, dtype=downsampled_data.dtype)

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
