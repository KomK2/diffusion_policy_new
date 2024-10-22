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

        ### Step 1: Get Previous 2 Points ###
        if data_index >= start + 2:
            prev_2_ft_data = ft_data_all[data_index-2:data_index]  # Previous 2 points
        else:
            # If there are less than 2 previous points, pad with zeros
            prev_2_ft_data = ft_data_all[start:data_index]
            missing_prev_points = 2 - prev_2_ft_data.shape[0]
            prev_padding = np.zeros((missing_prev_points, 10, 6))  # Padding with zeros
            prev_2_ft_data = np.concatenate((prev_padding, prev_2_ft_data), axis=0)

        ### Step 2: Get Next 2 Points ###
        if data_index < episode_end - 2:
            next_2_ft_data = ft_data_all[data_index+1:data_index+3]
        else:
            next_2_ft_data = ft_data_all[data_index+1:episode_end]
            missing_next_points = 2 - next_2_ft_data.shape[0]
            next_padding = np.zeros((missing_next_points, 10, 6))  # Padding with zeros
            next_2_ft_data = np.concatenate((next_2_ft_data, next_padding), axis=0)

        ### Step 3: Concatenate Previous 2 + Current + Next 2 ###
        current_ft_data = ft_data[np.newaxis, :, :]  # Shape (1, 10, 6)
        new_ft_data_point = np.concatenate((prev_2_ft_data, current_ft_data, next_2_ft_data), axis=0)

        # Reshape to combine all points into one array, resulting in a shape of (50, 6)
        new_ft_data_point = new_ft_data_point.reshape(-1, 6)
        assert new_ft_data_point.shape[0] == 50  # 2 previous, 1 current, and 2 next = 5 points, each with shape (10, 6) => (50, 6)
        
        # Add the final data point sequence to the list
        new_episodes_ft_data.append(new_ft_data_point)

    # Update start for the next episode
    start = episode_end

# Convert the list to a numpy array
new_episodes_ft_data = np.array(new_episodes_ft_data)

print(new_episodes_ft_data.shape)  # Ensure it has the expected shape
# Write the new dataset to the Zarr file under a new group 'data/new_ft_data'
z.create_dataset('data/new_ft_data', data=new_episodes_ft_data, shape=new_episodes_ft_data.shape, chunks=True, overwrite=True)

print("New dataset saved to Zarr.")

# # print("First 10 points of the first episode in ft_data_all:", ft_data_all[0:10]) 
# print("First 10 points of the first episode in new_episodes_ft_data:", new_episodes_ft_data[0].shape)
