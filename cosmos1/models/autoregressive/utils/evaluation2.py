import os
import glob
import h5py
import numpy as np
import torch
from einops import rearrange
import time
from tqdm import tqdm
from well_utils.data_processing.visualizations import create_video_heatmap
from cosmos1.models.autoregressive.utils.inference import resize_input

NUM_CONTEXT_FRAMES = int(os.environ["COSMOS_AR_NUM_TOTAL_FRAMES"])


import os
import glob
import h5py
import numpy as np
import torch
from einops import rearrange

class HDF5DataLoader:
    def __init__(self, path, num_input_frames, num_total_frames, data_resolution=None, device='cuda'):
        self.path = path
        self.num_input_frames = num_input_frames
        self.num_total_frames = num_total_frames
        self.data_resolution = data_resolution
        self.device = device
        self.file_paths = glob.glob(os.path.join(path, '*.hdf5'))
        
        if not self.file_paths:
            raise ValueError(f"No HDF5 files found in {path}")
            
    def get_generators(self):
        """Returns a generator of indexable file sample objects"""
        for file_path in tqdm(self.file_paths):
            yield HDF5FileSamples(
                file_path=file_path,
                num_input_frames=self.num_input_frames,
                num_total_frames=self.num_total_frames,
                data_resolution=self.data_resolution,
                device=self.device
            )

class HDF5FileSamples:
    def __init__(self, file_path, num_input_frames, num_total_frames, data_resolution, device):
        self.file_path = file_path
        self.num_input_frames = num_input_frames
        self.num_total_frames = num_total_frames
        self.resizer = lambda x: resize_input(x, data_resolution) if data_resolution is not None else x
        self.device = device

        with h5py.File(file_path, 'r') as f:
            self.video = f['data'][:]  # Shape is [T, H, W, C]

        T = self.video.shape[0]
        if T < self.num_total_frames:
            raise ValueError(f"File {file_path} has {T} frames but expected at least {self.num_total_frames}")
        self.num_samples = T - self.num_total_frames + 1

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.num_samples:
            raise IndexError(f"Index {idx} out of range for {self.num_samples} samples")

        i = idx  # Starting frame index for this sample

        # Process input frames
        input_frames = self.video[i:i+self.num_input_frames]
        last_frame = input_frames[-1:]
        repeated_frames = np.repeat(last_frame, self.num_total_frames - self.num_input_frames, axis=0)
        processed_input = np.concatenate([input_frames, repeated_frames], axis=0)

        # Convert to tensor and rearrange dimensions
        input_tensor = torch.from_numpy(processed_input)
        target_tensor = torch.from_numpy(self.video[i:i+self.num_total_frames])

        # Apply resizing if needed
        input_tensor = self.resizer(rearrange(input_tensor, 't h w c -> t c h w'))
        target_tensor = self.resizer(rearrange(target_tensor, 't h w c -> t c h w'))

        # Add batch dimension
        input_tensor = rearrange(input_tensor, 't c h w -> 1 c t h w')
        target_tensor = rearrange(target_tensor, 't c h w -> 1 c t h w')

        # Move to device if needed
        if self.device:
            input_tensor = input_tensor.to(self.device, non_blocking=True)
            target_tensor = target_tensor.to(self.device, non_blocking=True)

        return (input_tensor, target_tensor)