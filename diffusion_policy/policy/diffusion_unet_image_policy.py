from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
import torchvision

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from diffusion_policy.common.pytorch_util import dict_apply

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D


class DiffusionUnetImagePolicy(BaseImagePolicy):
    def __init__(self, 
            shape_meta: dict,
            noise_scheduler: DDPMScheduler,
            obs_encoder: MultiImageObsEncoder,
            horizon, 
            n_action_steps, 
            n_obs_steps,
            num_inference_steps=None,
            obs_as_global_cond=True,
            diffusion_step_embed_dim=256,
            down_dims=(256,512,1024),
            kernel_size=5,
            n_groups=8,
            cond_predict_scale=True,
            # parameters passed to step
            **kwargs):
        super().__init__()

        # parse shapes
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        # get feature dim
        obs_feature_dim = obs_encoder.output_shape()[0]

        # create diffusion model
        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        if obs_as_global_cond:
            input_dim = action_dim
            global_cond_dim = obs_feature_dim * n_obs_steps

        model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale
        )

        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps
    
    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, condition_mask,
            local_cond=None, global_cond=None,
            generator=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        
        # all_trajectories = []

        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)
        
        all_trajectories = [trajectory.clone().cpu().numpy()]
    
        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output
            model_output = model(trajectory, t, 
                local_cond=local_cond, global_cond=global_cond)

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, 
                generator=generator,
                **kwargs
                ).prev_sample
            
            unnormalized_trajectory = self.normalizer['action'].unnormalize(trajectory.clone())
            all_trajectories.append(unnormalized_trajectory.clone().detach().cpu().numpy())

        
        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]        

        return trajectory, all_trajectories


    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        assert 'past_action' not in obs_dict # not implemented yet
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        if self.obs_as_global_cond:
            # condition through global feature
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, Do
            global_cond = nobs_features.reshape(B, -1)
            # empty data for action
            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(B, To, -1)
            cond_data = torch.zeros(size=(B, T, Da+Do), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs_features
            cond_mask[:,:To,Da:] = True

        # run sampling
        nsample, all_trajectories = self.conditional_sample(
            cond_data, 
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            **self.kwargs)
        
        # unnormalize prediction
        naction_pred = nsample[...,:Da]

        # self.plot_denoising_process(all_trajectories, horizon_index=0, dimension_index=0)
        all_trajectories= np.array(all_trajectories)
        self.plot_denoising_process_3d(all_trajectories)


        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:,start:end]
        
        result = {
            'action': action,
            'action_pred': action_pred
        }
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        # normalize input
        assert 'valid_mask' not in batch
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = nactions
        cond_data = trajectory
        if self.obs_as_global_cond:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, 
                lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
            
            # camera_features = this_nobs['camera_0']
            # # Save images in camera_features
            # for i, img in enumerate(camera_features):
            #     img_path = f"/home/bmv/Kiran/img/image_{i}.png"
            #     torchvision.utils.save_image(img, img_path)

            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, Do
            global_cond = nobs_features.reshape(batch_size, -1)
        else:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory = cond_data.detach()

        # generate impainting mask
        condition_mask = self.mask_generator(trajectory.shape)

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (bsz,), device=trajectory.device
        ).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)
        
        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = cond_data[condition_mask]
        
        # Predict the noise residual
        pred = self.model(noisy_trajectory, timesteps, 
            local_cond=local_cond, global_cond=global_cond)
        
        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        return loss
    
    # def plot_denoising_process_3d(self, all_trajectories, save_path=None):
    #     """
    #     Animates how the prediction points (x, y, z) across all batches evolve in 3D over time,
    #     showing the denoising process.

    #     :param all_trajectories: Array with shape (16, 16, 16, 6) capturing the denoising process.
    #     :param save_path: File path to save the animation as a GIF (optional).
    #     """
        
    #     # Extract the x, y, z dimensions (0, 1, 2) across all batches
    #     # Shape: (denoising steps, batch_size * prediction_horizon, 3)
    #     trajectories = np.array([step[:, :, :3].reshape(-1, 3) for step in all_trajectories])

    #     fig = plt.figure()
    #     ax = fig.add_subplot(111, projection='3d')

    #     # Set axis limits based on the overall data range
    #     ax.set_xlim(np.min(trajectories[..., 0]), np.max(trajectories[..., 0]))
    #     ax.set_ylim(np.min(trajectories[..., 1]), np.max(trajectories[..., 1]))
    #     ax.set_zlim(np.min(trajectories[..., 2]), np.max(trajectories[..., 2]))

    #     ax.set_title('Denoising Progress')

    #     # Function to update the scatter plot for each frame in the animation
    #     def update(frame):
    #         ax.clear()  # Clear the plot for the next frame

    #         # Get the x, y, z coordinates for the current denoising step
    #         x_data = trajectories[frame][:, 0]  # x-coordinates for all batches and points
    #         y_data = trajectories[frame][:, 1]  # y-coordinates for all batches and points
    #         z_data = trajectories[frame][:, 2]  # z-coordinates for all batches and points

    #         # Plot the current prediction points as a scatter plot
    #         ax.scatter(x_data, y_data, z_data, color='blue', label=f'Step {frame}')

    #         # Set axis limits
    #         ax.set_xlim(np.min(trajectories[..., 0]), np.max(trajectories[..., 0]))
    #         ax.set_ylim(np.min(trajectories[..., 1]), np.max(trajectories[..., 1]))
    #         ax.set_zlim(np.min(trajectories[..., 2]), np.max(trajectories[..., 2]))

    #         ax.set_title(f'Denoising Progress - Step {frame}')
    #         ax.legend()

    #     # Create the animation
    #     anim = FuncAnimation(fig, update, frames=len(all_trajectories), interval=200)

    #     # Save or display the animation
    #     if save_path:
    #         anim.save(save_path, writer='imagemagick')
    #     else:
    #         plt.show()


    # def plot_denoising_process_3d(self, all_trajectories, save_path=None):
    #     """
    #     Animates how the prediction points (x, y, z) across all batches evolve in 3D over time,
    #     showing the denoising process.

    #     :param all_trajectories: Array with shape (16, 16, 16, 6) capturing the denoising process.
    #     :param save_path: File path to save the animation as a GIF (optional).
    #     """
        
    #     # Extract the x, y, z dimensions (0, 1, 2) across all batches
    #     # Shape: (denoising steps, batch_size * prediction_horizon, 3)
    #     trajectories = np.array([step[:, :, :3].reshape(-1, 3) for step in all_trajectories])

    #     fig = plt.figure()
    #     ax = fig.add_subplot(111, projection='3d')

    #     # Initialize the scatter plot with the first denoising step
    #     scatter = ax.scatter(trajectories[0][:, 0], trajectories[0][:, 1], trajectories[0][:, 2], color='blue')

    #     # Set axis limits based on the overall data range
    #     ax.set_xlim(np.min(trajectories[..., 0]), np.max(trajectories[..., 0]))
    #     ax.set_ylim(np.min(trajectories[..., 1]), np.max(trajectories[..., 1]))
    #     ax.set_zlim(np.min(trajectories[..., 2]), np.max(trajectories[..., 2]))

    #     ax.set_title('Denoising Progress')

    #     # Function to update the scatter plot for each frame in the animation
    #     def update(frame):
    #         # Update the x, y, z coordinates for the current denoising step
    #         x_data = trajectories[frame][:, 0]  # x-coordinates for all batches and points
    #         y_data = trajectories[frame][:, 1]  # y-coordinates for all batches and points
    #         z_data = trajectories[frame][:, 2]  # z-coordinates for all batches and points
            
    #         # This line updates the scatter plot's points to the current step
    #         scatter._offsets3d = (x_data, y_data, z_data)

    #         ax.set_title(f'Denoising Progress - Step {frame}')

    #     # Ensure the animation goes forward without resetting
    #     anim = FuncAnimation(fig, update, frames=len(all_trajectories), interval=200, repeat=False)

    #     # Save or display the animation
    #     if save_path:
    #         anim.save(save_path, writer='imagemagick')
    #     else:
    #         plt.show()

    def plot_denoising_process_3d(self, all_trajectories, save_path=None):
        """
        Animates how the prediction points (x, y, z) across all batches evolve in 3D over time,
        showing the denoising process.

        :param all_trajectories: Array with shape (16, 16, 16, 6) capturing the denoising process.
        :param save_path: File path to save the animation as a GIF (optional).
        """
        
        # Extract the x, y, z dimensions (0, 1, 2) across all batches
        # Shape: (denoising steps, batch_size * prediction_horizon, 3)
        trajectories = np.array([step[:, :, :3].reshape(-1, 3) for step in all_trajectories])

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Initialize the scatter plot with the first denoising step
        scatter = ax.scatter(trajectories[0][:, 0], trajectories[0][:, 1], trajectories[0][:, 2], color='blue')

        # Set axis limits based on the overall data range
        ax.set_xlim(np.min(trajectories[..., 0]), np.max(trajectories[..., 0]))
        ax.set_ylim(np.min(trajectories[..., 1]), np.max(trajectories[..., 1]))
        ax.set_zlim(np.min(trajectories[..., 2]), np.max(trajectories[..., 2]))

        ax.set_title('Denoising Progress')

        # Define a list of colors for each diffusion step
        colors = plt.cm.viridis(np.linspace(0, 1, len(all_trajectories)))  # Using a colormap for gradient colors

        # Function to update the scatter plot for each frame in the animation
        def update(frame):
            # Update the x, y, z coordinates for the current denoising step
            x_data = trajectories[frame][:, 0]  # x-coordinates for all batches and points
            y_data = trajectories[frame][:, 1]  # y-coordinates for all batches and points
            z_data = trajectories[frame][:, 2]  # z-coordinates for all batches and points
            
            # Update the scatter plot's points and color
            scatter._offsets3d = (x_data, y_data, z_data)
            scatter.set_color(colors[frame])  # Change color based on the diffusion step

            ax.set_title(f'Denoising Progress - Step {frame}')

        # Ensure the animation goes forward without resetting
        anim = FuncAnimation(fig, update, frames=len(all_trajectories), interval=200, repeat=False)

        # Save or display the animation
        if save_path:
            anim.save(save_path, writer='imagemagick')
        else:
            plt.show()
