import zarr
import numpy as np
import matplotlib.pyplot as plt
from viz_constants import VIZ_DIR

dataset_dir = VIZ_DIR
# Load the Zarr dataset
z = zarr.open(f'/home/bmv/diffusion_policy_new/data/real_crab_85/replay_buffer.zarr', mode='r')

ep_len = z['meta/episode_ends'][:]
ft = z['data/ft_data'][:]

def plot_per_episode_ft(ax, episode_data):
    # Plot ft data for a specific episode on a given axis (subplot)
    time_steps = np.arange(len(episode_data)) / 10.0  # Divide by 10 to convert to seconds
    ax.plot(time_steps, episode_data)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Force/Torque')
    ax.grid(True)

def visualize_all_episodes_ft(ft_data, ep_len, num_cols=5):
    num_episodes = len(ep_len)
    num_rows = (num_episodes + num_cols - 1) // num_cols  # Calculate number of rows based on number of episodes and columns
    print(f"Total number of episodes: {num_episodes}")

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))

    for episode_index in range(num_episodes):
        row = episode_index // num_cols
        col = episode_index % num_cols
        ax = axs[row, col] if num_rows > 1 else axs[col]
        
        start_index = 0 if episode_index == 0 else ep_len[episode_index - 1]
        end_index = ep_len[episode_index]
        episode_data = ft_data[start_index:end_index]
        
        plot_per_episode_ft(ax, episode_data)

    # Adjust layout and spacing
    fig.tight_layout(pad=0.5)
    plt.show()

# Function to plot ft data for a particular episode
def plot_episode_ft(ft_data, ep_len, episode_index):
    # Plot ft data for a specific episode
    start_index = 0 if episode_index == 0 else ep_len[episode_index - 1]
    end_index = ep_len[episode_index]
    episode_ft_data = ft_data[start_index:end_index]

    time_steps = np.arange(len(episode_ft_data)) / 50.0  # Divide by 10 to convert to seconds

    plt.figure(figsize=(10, 6))
    plt.plot(time_steps, episode_ft_data)
    plt.title(f'Force/Torque Data')
    plt.xlabel('Time (s)')
    plt.ylabel('Force(N) /Torque (N-m)')
    plt.legend(['FX', 'FY', 'FZ', 'TX', 'TY', 'TZ'])  # Assuming order of dimensions
    plt.grid(True)
    plt.show()

print(f"Visualizing Dataset: {dataset_dir}")
episode_index = int(input("Enter episode index to visualize || -1 for all data: "))
if episode_index >= 0:
    plot_episode_ft(ft, ep_len, episode_index=episode_index)
elif episode_index == -1:
    visualize_all_episodes_ft(ft, ep_len)
else:
    raise NotImplementedError
