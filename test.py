import zarr
import numpy as np

z = zarr.open(f'/home/bmv/diffusion_policy_new/data/ft_vision/replay_buffer.zarr', mode='r')

episode_ends = z['meta/episode_ends'][:]
ft_data_all = z['data/ft_data'][:]

start = 0
new_episodes_ft_data = []
for indx , ep_ind in enumerate(episode_ends):
    if indx == 0:
        start = 0
    end = ep_ind

    
    for data_index in range(start,end):
        ft_data = ft_data_all[data_index]
        
        if (data_index < end - 1 -5):
            next_5_ft_data = ft_data_all[data_index+1:data_index+6]  

            ft_data_point = ft_data[np.newaxis, :, :]
            new_ft_data_point = np.concatenate((ft_data_point, next_5_ft_data), axis=0)
            
            
        elif (data_index >= end -5) and data_index < end :
            next_5_ft_data = ft_data_all[data_index+1:end]
            ft_data_point = ft_data[np.newaxis, :, :]

            missing_points = 5 - next_5_ft_data.shape[0]
            padding = np.zeros((missing_points,10,6))

            new_ft_data_point = np.concatenate((ft_data_point, next_5_ft_data, padding), axis=0)

        assert new_ft_data_point.shape[0] == 6

        new_episodes_ft_data.append(new_ft_data_point)
            
    
    start = end

    

    
