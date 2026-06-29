import os
import cv2

def save_episodes_videos(videos, output_video_dir, fps: float = 30):
    """
    Saves videos of the episodes in the folder output_video_dir
    with the names episode_i.mp4 for video of index i.
    """
    os.makedirs(output_video_dir, exist_ok=True)
    for i, video in enumerate(videos):
        output_video_path = os.path.join(output_video_dir, f'episode_{i}.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for .mp4 file

        # Get frame size dynamically from the first image in your list
        first_image = video[0]
        height, width, layers = first_image.shape
        frame_size = (width, height) # OpenCV expects (width, height)

        # --- 2. Initialize VideoWriter ---
        out = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)

        # --- 3. & 4. Process Your Image List and Write Frames ---
        for image_array in video:
            # IMPORTANT: Convert the RGB image (standard in many libraries)
            # to BGR (what OpenCV uses)
            bgr_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            out.write(bgr_image)

        # --- 5. Release everything when the job is finished ---
        out.release()
        print(f"Video saved successfully to: {output_video_path}")
        cv2.destroyAllWindows() # This is good practice if you were displaying windows

def save_episodes_frames(videos, output_dir, fps: float = 30, image_format: str = "png",
                         zero_pad: int = 6, start_index: int = 0):
    """
    Saves each episode (a sequence of RGB images) into its own folder under output_dir.
    Frames are named frame_XXXXXX.<ext> with zero padding for easy sorting.

    Args:
        videos: Iterable of episodes, where each episode is an iterable of HxWx3 RGB numpy arrays.
        output_dir (str): Root directory where episode folders are created (episode_0, episode_1, ...).
        fps (float): Optional. Only stored to a text file for reference (no effect on images).
        image_format (str): File extension/format for frames (e.g., "png", "jpg").
        zero_pad (int): Number of digits to left-pad the frame index (default 6).
        start_index (int): Starting index for frame numbering within each episode (default 0).
    """
    os.makedirs(output_dir, exist_ok=True)

    for i, video in enumerate(videos):
        episode_dir = os.path.join(output_dir, f"episode_{i}")
        os.makedirs(episode_dir, exist_ok=True)

        # Save FPS info for reference
        with open(os.path.join(episode_dir, "fps.txt"), "w") as f:
            f.write(str(fps))

        for t, img in enumerate(video, start=start_index):
            if img is None:
                continue  # skip empty frames just in case

            # OpenCV expects BGR, so convert from RGB
            bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # Build filename
            fname = f"frame_{t:0{zero_pad}d}.{image_format.lower()}"
            fpath = os.path.join(episode_dir, fname)

            ok = cv2.imwrite(fpath, bgr)
            if not ok:
                raise IOError(f"Failed to write frame to {fpath}")

        print(f"Frames for episode {i} saved to: {episode_dir}")