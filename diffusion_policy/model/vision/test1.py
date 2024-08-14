from typing import Dict, Tuple, Union
import copy
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torchvision
from diffusion_policy.model.vision.crop_randomizer import CropRandomizer
from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin
from diffusion_policy.common.pytorch_util import dict_apply, replace_submodules


class MultiImageObsEncoder(ModuleAttrMixin):
    def __init__(self,
                 shape_meta: dict,
                 rgb_model: Union[nn.Module, Dict[str,nn.Module]],
                 resize_shape: Union[Tuple[int,int], Dict[str,tuple], None]=None,
                 crop_shape: Union[Tuple[int,int], Dict[str,tuple], None]=None,
                 random_crop: bool=True,
                 use_group_norm: bool=False,
                 share_rgb_model: bool=False,
                 imagenet_norm: bool=False):
        super().__init__()

        self.num = 0  # Counter to control when to visualize
        self.intermediate_features = {}

        rgb_keys = list()
        low_dim_keys = list()
        key_model_map = nn.ModuleDict()
        key_transform_map = nn.ModuleDict()
        key_shape_map = dict()

        if share_rgb_model:
            assert isinstance(rgb_model, nn.Module)
            key_model_map['rgb'] = rgb_model

        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            type = attr.get('type', 'low_dim')
            key_shape_map[key] = shape
            if type == 'rgb':
                rgb_keys.append(key)
                this_model = None
                if not share_rgb_model:
                    if isinstance(rgb_model, dict):
                        this_model = rgb_model[key]
                    else:
                        assert isinstance(rgb_model, nn.Module)
                        this_model = copy.deepcopy(rgb_model)
                
                if this_model is not None:
                    if use_group_norm:
                        this_model = replace_submodules(
                            root_module=this_model,
                            predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                            func=lambda x: nn.GroupNorm(
                                num_groups=x.num_features//16, 
                                num_channels=x.num_features)
                        )
                    key_model_map[key] = this_model
                
                input_shape = shape
                this_resizer = nn.Identity()
                if resize_shape is not None:
                    if isinstance(resize_shape, dict):
                        h, w = resize_shape[key]
                    else:
                        h, w = resize_shape
                    this_resizer = torchvision.transforms.Resize(size=(h, w))
                    input_shape = (shape[0], h, w)

                this_randomizer = nn.Identity()
                if crop_shape is not None:
                    if isinstance(crop_shape, dict):
                        h, w = crop_shape[key]
                    else:
                        h, w = crop_shape
                    if random_crop:
                        this_randomizer = CropRandomizer(
                            input_shape=input_shape,
                            crop_height=h,
                            crop_width=w,
                            num_crops=1,
                            pos_enc=False
                        )
                    else:
                        this_randomizer = torchvision.transforms.CenterCrop(size=(h, w))
                this_normalizer = nn.Identity()
                if imagenet_norm:
                    this_normalizer = torchvision.transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                
                this_transform = nn.Sequential(this_resizer, this_randomizer, this_normalizer)
                key_transform_map[key] = this_transform
            elif type == 'low_dim':
                low_dim_keys.append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")
        rgb_keys = sorted(rgb_keys)
        low_dim_keys = sorted(low_dim_keys)

        self.shape_meta = shape_meta
        self.key_model_map = key_model_map
        model_without_fc = nn.Sequential(*list(key_model_map["camera_1"].children())[:-5])

# Apply the image to this modified model
        # intermediate_features = model_without_fc(img)
        self.model_without_fc = model_without_fc
        self.key_transform_map = key_transform_map
        self.share_rgb_model = share_rgb_model
        self.rgb_keys = rgb_keys
        self.low_dim_keys = low_dim_keys
        self.key_shape_map = key_shape_map

    # def _register_hooks(self, model):
    #     """Register hooks to capture intermediate features."""
    #     def hook(module, input, output, name):
    #         self.intermediate_features[name] = output

    #     for name, layer in model.named_modules():
    #         if isinstance(layer, (nn.Conv2d, nn.ReLU, nn.MaxPool2d, nn.AdaptiveAvgPool2d)):
    #             layer.register_forward_hook(lambda module, input, output, layer_name=name: hook(module, input, output, layer_name))

    def forward(self, obs_dict):
        batch_size = None
        features = list()
        feature_map = dict()

        for key in self.rgb_keys:
            model = self.key_model_map[key]
            # self._register_hooks(model)

        # process rgb input
        if self.share_rgb_model:
            imgs = list()
            for key in self.rgb_keys:
                img = obs_dict[key]
                if batch_size is None:
                    batch_size = img.shape[0]
                else:
                    assert batch_size == img.shape[0]
                assert img.shape[1:] == self.key_shape_map[key]
                img = self.key_transform_map[key](img)
                imgs.append(img)
            imgs = torch.cat(imgs, dim=0)
            feature = self.key_model_map['rgb'](imgs)
            feature = feature.reshape(-1, batch_size, *feature.shape[1:])
            feature = torch.moveaxis(feature, 0, 1)
            feature = feature.reshape(batch_size, -1)
            features.append(feature)
        else:
            for key in self.rgb_keys:
                img = obs_dict[key]
                if batch_size is None:
                    batch_size = img.shape[0]
                else:
                    assert batch_size == img.shape[0]
                assert img.shape[1:] == self.key_shape_map[key]
                img = self.key_transform_map[key](img)
                feature = self.key_model_map[key](img)
                feature_without_fc = self.model_without_fc(img)
                
                features.append(feature)

                if self.num == 1:
                    feature_map[key] = feature
                    print(f"Visualizing features from {key} with shape {feature_without_fc.shape}")

                    self.visualize_feature_map(feature_without_fc)
                self.num += 1

        for key in self.low_dim_keys:
            data = obs_dict[key]
            if batch_size is None:
                batch_size = data.shape[0]
            else:
                assert batch_size == data.shape[0]
            assert data.shape[1:] == self.key_shape_map[key]
            features.append(data)

        result = torch.cat(features, dim=-1)
        return result
    
    def visualize_feature_map(slef , feature_map):
   
   

        feature_map = feature_map.squeeze(0)
        print(f"Feature map shape: {feature_map.shape}")

        # Number of channels to visualize (e.g., 6 or 8)
        num_channels_to_visualize = min(8, feature_map.shape[0])  # Visualize up to 8 channels
        fig, axs = plt.subplots(1, num_channels_to_visualize, figsize=(20, 5))

        for i in range(num_channels_to_visualize):
            axs[i].imshow(feature_map[i].detach().cpu().numpy(), cmap='viridis')
            axs[i].axis('off')
            axs[i].set_title(f'Channel {i + 1}')

        plt.show()

    



    @torch.no_grad()
    def output_shape(self):
        example_obs_dict = dict()
        obs_shape_meta = self.shape_meta['obs']
        batch_size = 1
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            this_obs = torch.zeros(
                (batch_size,) + shape, 
                dtype=self.dtype,
                device=self.device)
            example_obs_dict[key] = this_obs
        example_output = self.forward(example_obs_dict)
        output_shape = example_output.shape[1:]
        return output_shape
