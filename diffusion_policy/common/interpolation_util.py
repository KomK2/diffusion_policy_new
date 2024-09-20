import numpy as np
import scipy.interpolate as si
import scipy.spatial.transform as st


def get_interp1d(t, x):
    gripper_interp = si.interp1d(
        t, x, 
        axis=0, bounds_error=False, 
        fill_value=(x[0], x[-1]))
    return gripper_interp

def interpolate_ft_data(ft_timestamps, ft_data, target_timestamps):
    interpolator = get_interp1d(ft_timestamps, ft_data)
    return interpolator(target_timestamps)

class PoseInterpolator:
    def __init__(self, t, x):
        pos = x[:,:3]
        rot = st.Rotation.from_rotvec(x[:,3:])
        self.pos_interp = get_interp1d(t, pos)
        self.rot_interp = st.Slerp(t, rot)
    
    @property
    def x(self):
        return self.pos_interp.x
    
    def __call__(self, t):
        min_t = self.pos_interp.x[0]
        max_t = self.pos_interp.x[-1]
        t = np.clip(t, min_t, max_t)

        pos = self.pos_interp(t)
        rot = self.rot_interp(t)
        rvec = rot.as_rotvec()
        pose = np.concatenate([pos, rvec], axis=-1)
        return pose
    
def linear_interpolate_zeros(data):
    """
    Interpolates rows in the data array that are all zeros using linear interpolation.
    """
    non_zero_indices = np.where(~np.all(data == 0, axis=1))[0]
    zero_indices = np.where(np.all(data == 0, axis=1))[0]

    for zero_idx in zero_indices:
        # Find the closest non-zero indices before and after the zero_idx
        prev_non_zero_idx = non_zero_indices[non_zero_indices < zero_idx]
        next_non_zero_idx = non_zero_indices[non_zero_indices > zero_idx]

        if len(prev_non_zero_idx) == 0 or len(next_non_zero_idx) == 0:
            continue  # Skip interpolation if there are no valid surrounding rows

        prev_non_zero_idx = prev_non_zero_idx.max()
        next_non_zero_idx = next_non_zero_idx.min()
        
        # Perform linear interpolation
        weight = (zero_idx - prev_non_zero_idx) / (next_non_zero_idx - prev_non_zero_idx)
        data[zero_idx] = (1 - weight) * data[prev_non_zero_idx] + weight * data[next_non_zero_idx]
    
    return data
