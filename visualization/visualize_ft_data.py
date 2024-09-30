import zarr
import numpy as np
import matplotlib.pyplot as plt
from viz_constants import VIZ_DIR

dataset_dir = VIZ_DIR
# Load the Zarr dataset
z = zarr.open(f'/home/bmv/diffusion_policy_new/data/ft_100hz/replay_buffer.zarr', mode='r')

ep_len = z['meta/episode_ends'][:]
ft = z['data/ft_data'][:]
print("ft data shape:", ft.shape)

# def plot_per_episode_ft(ax, episode_data):
#     # Reshape the data to account for frequency (flatten it)
#     reshaped_data = episode_data.reshape(-1, 6)
#     # Plot ft data for a specific episode on a given axis (subplot)
#     ax.plot(reshaped_data)
#     ax.grid(True)


def plot_per_episode_ft(ax, episode_data):
    # Reshape the data to account for frequency (flatten it)
    reshaped_data = episode_data.reshape(-1, 6)
    time_steps = np.arange(reshaped_data.shape[0])  # Create time steps corresponding to each frequency row
    
    # Plot only the Z-force (FZ) which is the 3rd column
    ax.scatter(time_steps, reshaped_data[:, 2], label='FZ', s=10)  # s=10 sets the marker size
    
    ax.grid(True)
    ax.legend(fontsize=8)  # Add legend for Z-force

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
        
        # Call the updated function to plot only Z-force for each episode
        plot_per_episode_ft(ax, episode_data)

    # Adjust layout and spacing
    fig.tight_layout(pad=0.5)
    plt.show()


# def visualize_all_episodes_ft(ft_data, ep_len, num_cols=5):
#     num_episodes = len(ep_len)
#     num_rows = (num_episodes + num_cols - 1) // num_cols  # Calculate number of rows based on number of episodes and columns
#     print(f"Total number of episodes: {num_episodes}")

#     fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))

#     for episode_index in range(num_episodes):
#         row = episode_index // num_cols
#         col = episode_index % num_cols
#         ax = axs[row, col] if num_rows > 1 else axs[col]
        
#         start_index = 0 if episode_index == 0 else ep_len[episode_index - 1]
#         end_index = ep_len[episode_index]
#         episode_data = ft_data[start_index:end_index]
        
#         plot_per_episode_ft(ax, episode_data)

#     # Adjust layout and spacing
#     fig.tight_layout(pad=0.5)
#     plt.show()

# Function to plot ft data for a particular episode
def plot_episode_ft(ft_data, ep_len, episode_index):
    # Plot ft data for a specific episode
    start_index = 0 if episode_index == 0 else ep_len[episode_index - 1]
    end_index = ep_len[episode_index]
    episode_ft_data = ft_data[start_index:end_index]
    
    # Reshape the data to account for frequency
    episode_ft_data = episode_ft_data.reshape(-1, 6)
    
    with np.printoptions(threshold=np.inf):
        print("episode_ft_data:", episode_ft_data)
    
    plt.figure(figsize=(10, 6))
    time_steps = np.arange(episode_ft_data.shape[0])  # Create time steps corresponding to each frequency row

    # Plot scatter for each force/torque dimension
    for i, label in enumerate(['FX', 'FY', 'FZ', 'TX', 'TY', 'TZ']):
        plt.scatter(time_steps, episode_ft_data[:, i], label=label)
    
    plt.title(f'Force/Torque Data for Episode {episode_index} (Scatter Plot)')
    plt.xlabel('Time Step')
    plt.ylabel('Force/Torque')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_all_ft_overlapped(ft_data, ep_len):
    plt.figure(figsize=(10, 6))
    
    num_episodes = len(ep_len)
    for episode_index in range(num_episodes):
        if num_episodes//2  <= episode_index :
            break
            
        start_index = 0 if episode_index == 0 else ep_len[episode_index - 1]
        end_index = ep_len[episode_index]
        episode_ft_data = ft_data[start_index:end_index]
        
        # Reshape the data to account for frequency
        episode_ft_data = episode_ft_data.reshape(-1, 6)
        time_steps = np.arange(episode_ft_data.shape[0])  # Create time steps for x-axis
        
        # Plot only the Z-force (FZ) for all episodes
        plt.plot(time_steps, episode_ft_data[:, 2], label=f'Episode {episode_index}', alpha=0.5)  # Use alpha to show overlap

    plt.title('Overlapped Z-Force (FZ) for All Episodes')
    plt.xlabel('Time Step')
    plt.ylabel('Z-Force (FZ)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


print(f"Visualizing Dataset: {dataset_dir}")
episode_index = int(input("Enter episode index to visualize || -1 for all data: "))
if episode_index >= 0:
    plot_episode_ft(ft, ep_len, episode_index=episode_index)
elif episode_index == -1:
    plot_all_ft_overlapped(ft, ep_len)
else:
    raise NotImplementedError
