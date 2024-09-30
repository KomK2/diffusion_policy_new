import os
import subprocess
import concurrent.futures

# Original and new directories
ORIGINAL_DIR = "/home/bmv/diffusion_policy_new/data/replicaless_rice_scoop/videos"
NEW_DIR = "/home/bmv/diffusion_policy_new/data/replicaless_rice_scoop/new_videos"

# Brightness factor
BRIGHTNESS_FACTOR = "0.1"

def process_video(args):
    original_file_path, new_file_path = args
    print(f"Processing: {original_file_path}")
    subprocess.run([
        "ffmpeg", "-y", "-i", original_file_path,
        "-vf", f"eq=brightness={BRIGHTNESS_FACTOR}",
        new_file_path
    ])

video_files = []

# Collect all video file paths
for root, dirs, files in os.walk(ORIGINAL_DIR):
    relative_path = os.path.relpath(root, ORIGINAL_DIR)
    new_root = os.path.join(NEW_DIR, relative_path)
    os.makedirs(new_root, exist_ok=True)
    for file in files:
        if file.endswith(".mp4"):
            original_file_path = os.path.join(root, file)
            new_file_path = os.path.join(new_root, file)
            video_files.append((original_file_path, new_file_path))

# Process videos in parallel
with concurrent.futures.ThreadPoolExecutor() as executor:
    executor.map(process_video, video_files)
