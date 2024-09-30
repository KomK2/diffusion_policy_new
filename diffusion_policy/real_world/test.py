import cv2
import numpy as np

# Path to the image
image_path = '/home/bmv/Kiran/episode_0_frame_0_camera_0.jpg'

# Read the image
image = cv2.imread(image_path)

# Create a mask with the same dimensions as the image, initialized to all black (0)
mask = np.zeros_like(image)

# Specify the region you want to keep (cropped_part)
cropped_part = image[180:, 70:]

# Apply the cropped part to the mask
mask[180:, 70:] = cropped_part

# Save the new image with the masked region
output_path = '/home/bmv/Kiran/cropped_image.jpg'
cv2.imwrite(output_path, mask)

print(f'Cropped image saved at {output_path}')
