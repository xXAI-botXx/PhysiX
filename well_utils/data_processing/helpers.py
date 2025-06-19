
import h5py
from pathlib import Path
import numpy as np
import cv2
import imutils

def save_numpy_as_mp4(frames, save_path, fps=30):
    num_frames, height, width, channels = frames.shape
    if channels != 3:
        raise ValueError("Frames must have 3 channels (RGB).")

    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

    for frame in frames:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)

    out.release()

def resize_video_array(video_array, width=None, height=None):
    # Initialize output array
    if width is not None:
        first_frame = imutils.resize(video_array[0], width=width)
    elif height is not None:
        first_frame = imutils.resize(video_array[0], height=height)
    else:
        return video_array
        
    resized_video = np.zeros((
        video_array.shape[0],
        first_frame.shape[0],
        first_frame.shape[1],
        video_array.shape[3]
    ))
    
    # Resize each frame
    for i in range(len(video_array)):
        if width is not None:
            resized_video[i] = imutils.resize(video_array[i], width=width)
        else:
            resized_video[i] = imutils.resize(video_array[i], height=height)
            
    return resized_video
