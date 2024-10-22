import zarr
import numpy as np

# Open Zarr file
z = zarr.open(f'/home/bmv/diffusion_policy_new/data/ft_vision/replay_buffer.zarr', mode='a')

# Load episode ends and ft_data
episode_ends = z['meta/episode_ends'][:]
ft_data_all = z['data/ft_data'][:]

start = 0
new_episodes_ft_data = []

# Loop through each episode
for episode_idx, episode_end in enumerate(episode_ends):
    start = 0 if episode_idx == 0 else episode_ends[episode_idx - 1]
    
    # Loop through data points in the current episode
    for data_index in range(start, episode_end):
        ft_data = ft_data_all[data_index]

        ### Step 1: Get Previous 4 Points ###
        if data_index >= start + 4:
            prev_4_ft_data = ft_data_all[data_index-4:data_index]  # Previous 4 points
        else:
            # If there are less than 4 previous points, pad with zeros
            prev_4_ft_data = ft_data_all[start:data_index]
            missing_prev_points = 4 - prev_4_ft_data.shape[0]
            prev_padding = np.zeros((missing_prev_points, 10, 6))  # Padding with zeros
            prev_4_ft_data = np.concatenate((prev_padding, prev_4_ft_data), axis=0)

        ### Step 2: Get Next 5 Points ###
        if data_index < episode_end - 5:
            next_5_ft_data = ft_data_all[data_index+1:data_index+6]
        else:
            next_5_ft_data = ft_data_all[data_index+1:episode_end]
            missing_next_points = 5 - next_5_ft_data.shape[0]
            next_padding = np.zeros((missing_next_points, 10, 6))  # Padding with zeros
            next_5_ft_data = np.concatenate((next_5_ft_data, next_padding), axis=0)

        ### Step 3: Concatenate Previous 4 + Current + Next 5 ###
        current_ft_data = ft_data[np.newaxis, :, :]  # Shape (1, 10, 6)
        new_ft_data_point = np.concatenate((prev_4_ft_data, current_ft_data, next_5_ft_data), axis=0)

        new_ft_data_point = new_ft_data_point.reshape(-1,6)
        assert new_ft_data_point.shape[0] == 100
        

        
        # Add the final data point sequence to the list
        new_episodes_ft_data.append(new_ft_data_point)

    # Update start for the next episode
    start = episode_end

new_episodes_ft_data = np.array(new_episodes_ft_data)

print(new_episodes_ft_data.shape)
# Write the new dataset to the Zarr file under a new group 'data/new_ft_data'
# z.create_dataset('data/new_ft_data', data=new_episodes_ft_data, shape=new_episodes_ft_data.shape, chunks=True, overwrite=True)

# print("New dataset saved to Zarr.")

print("first 10 points of the first episode", ft_data_all[0:10]) 
print("first 10 points of the first episode", new_episodes_ft_data[0])