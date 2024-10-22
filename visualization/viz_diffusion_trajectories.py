import os
import pathlib
import sys
import cv2
import dill
import hydra
import torch
from torch.utils.data import DataLoader
from threadpoolctl import threadpool_limits

# from diffusion_policy.dataset.real_pusht_image_dataset import zarr_resize_index_last_dim
# from diffusion_policy.real_world.real_data_conversion import real_data_to_replay_buffer


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.workspace.base_workspace import BaseWorkspace

# def _get_replay_buffer(dataset_path, shape_meta, store):
#     # parse shape meta
#     rgb_keys = list()
#     lowdim_keys = list()
#     out_resolutions = dict()
#     lowdim_shapes = dict()
#     obs_shape_meta = shape_meta['obs']
#     for key, attr in obs_shape_meta.items():
#         type = attr.get('type', 'low_dim')
#         shape = tuple(attr.get('shape'))
#         if type == 'rgb':
#             rgb_keys.append(key)
#             c,h,w = shape
#             out_resolutions[key] = (w,h)
#         elif type == 'low_dim':
#             lowdim_keys.append(key)
#             lowdim_shapes[key] = tuple(shape)
#             if 'pose' in key:
#                 assert tuple(shape) in [(2,),(6,)]
    
#     action_shape = tuple(shape_meta['action']['shape'])
#     assert action_shape in [(2,),(6,)]

#     # load data
#     cv2.setNumThreads(1)
#     with threadpool_limits(1):
#         replay_buffer = real_data_to_replay_buffer(
#             dataset_path=dataset_path,
#             out_store=store,
#             out_resolutions=out_resolutions,
#             lowdim_keys=lowdim_keys + ['action'],
#             image_keys=rgb_keys
#         )
    
#     for key, shape in lowdim_shapes.items():
#         if 'pose' in key and shape == (2,):
#             # only take X and Y
#             zarr_arr = replay_buffer[key]
#             zarr_resize_index_last_dim(zarr_arr, idxs=[0,1])

#     return replay_buffer

def load_checkpoint(checkpoint_path):
    payload = torch.load(open(checkpoint_path, 'rb'), pickle_module=dill )
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    print('==== checkpoint loaded ====')

    policy: BaseImagePolicy
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model

    device = torch.device('cuda')

    # policy.num_inference_steps = 15 # DDIM inference iterations
    # policy.n_action_steps = policy.horizon - policy.n_obs_steps + 1

    device = torch.device(cfg.training.device)


    dataset: BaseImageDataset
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    assert isinstance(dataset, BaseImageDataset)
    
    train_dataloader = DataLoader(dataset, **cfg.dataloader)
    normalizer = dataset.get_normalizer()

    count = 0
    for i, batch in enumerate(train_dataloader):

        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
        obs_dict = batch['obs']
                        
        result = policy.predict_action(obs_dict)
        
        # count += 1

        # if count > 10:
        #     break




if __name__ == '__main__':

    checkpoint_path = '/home/bmv/diffusion_policy_new/data/outputs/2024.10.16_ft_vision/14.34.33_train_diffusion_unet_image_ft_vision/checkpoints/latest.ckpt'
    load_checkpoint(checkpoint_path)
