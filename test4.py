import zarr
import numpy as np

# Load the Zarr dataset
z = zarr.open('/home/bmv/diffusion_policy_new/data/rice_scoop_teleop_2/replay_buffer.zarr', mode='r')

replica_poses = z['data/replica_eef_pose'][:]
action = z['data/action'][:]
timestamp = z['data/timestamp'][:]
stage = z['data/stage'][:]

ep_len = z['meta/episode_ends'][:]

new_replica_poses = []
new_actions = []
new_timestamp = []
new_stage = []
new_ep_len = []

start_index = 0
new_ep = 0

# loop through the episodes
for index,episode  in enumerate(ep_len):
 
    episode_index = index
    start_index = 0 if episode_index == 0 else ep_len[episode_index - 1]
    end_index = ep_len[episode_index]

    episode_length = 0

    # loop through each episode
    new_data = []
    for i in range(start_index,end_index):
        if np.all(replica_poses[i] == np.zeros(6)):
            continue
        
        new_replica_poses.append(replica_poses[i])
        new_actions.append(action[i])
        new_timestamp.append(timestamp[i])
        new_stage.append(stage[i])

        new_data.append(replica_poses[i])

        episode_length += 1 

    if len(new_ep_len) == 0:
        new_ep_len.append(episode_length)
    else:
        new_ep_len.append(new_ep_len[-1] + episode_length)

new_replica_poses = np.array(new_replica_poses)
new_actions = np.array(new_actions)
new_timestamp = np.array(new_timestamp)
new_stage = np.array(new_stage)
new_ep_len = np.array(new_ep_len)



new_start_index = 2
new_end_index = new_ep_len[2]

episode_index = 2
start_index = 0 if episode_index == 0 else ep_len[episode_index - 1]
end_index = ep_len[episode_index]

print("new end effector pose :", new_replica_poses[new_start_index:new_end_index].shape)
print("old end effector pose :", replica_poses[start_index:end_index].shape)


    
    





# episode_index = 2

# start_index = 0 if episode_index == 0 else ep_len[episode_index - 1]
# end_index = ep_len[episode_index]

# y = replica_poses[start_index:end_index][1]
# z = np.zeros(6)
# x = np.all(y == z)

# print("compare :" ,x  )


