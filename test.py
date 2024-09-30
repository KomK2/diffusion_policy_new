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

def process_ft_data(data, factor):
    processed_data = []
    for i in range(0, len(data) - factor + 1, factor):
        window = data[i:i + factor]
        processed_data.append(np.hstack(window))  
    return np.array(processed_data)

z = zarr.open(f'/home/bmv/diffusion_policy_new/data/real_crab_85/replay_buffer.zarr', mode='r')
episode_ends = z['meta/episode_ends'][:]

factor = 5  

# Create the new Zarr file for saving the processed data
output_zarr_path = '/home/bmv/diffusion_policy_new/data/real_crab_85_10hz_down_sampled/replay_buffer.zarr'
output_z = zarr.open(output_zarr_path, mode='w')
output_group = output_z.create_group('data')
meta_group = output_z.create_group('meta')

for name, item in z['data'].items():
    data = item[:]
    
    if name == 'ft_data':
        continue
        # processed_ft_data = process_ft_data(data, factor)
        # output_group.create_dataset(name, data=processed_ft_data, shape=processed_ft_data.shape, dtype=processed_ft_data.dtype)
    else:
        downsampled_data, downsample_end_index = downsample_episodes(data, episode_ends, factor)  # Downsample other data
        output_group.create_dataset(name, data=downsampled_data, shape=downsampled_data.shape, dtype=downsampled_data.dtype)

    # If needed, you can print the shapes for verification
    # print(f'{name} after processing:', processed_ft_data.shape if name == 'ft_data' else downsampled_data.shape)

# Save the new episode ends for the downsampled data
meta_group.create_dataset('episode_ends', data=downsample_end_index, shape=downsample_end_index.shape, dtype=downsample_end_index.dtype)

# Copy other metadata items from the original file
for name, item in z['meta'].items():
    if name != 'episode_ends':
        data = item[:]
        meta_group.create_dataset(name, data=data, shape=data.shape, chunks=True, dtype=data.dtype)

print("Processing complete")
