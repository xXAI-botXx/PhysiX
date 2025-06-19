import os
import random
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
import  numpy as np
from cosmos1.models.autoregressive.utils.inference import resize_input

class RandomHDF5Dataset(Dataset):
    def __init__(
        self,
        data_dir,
        n_frames,
        data_resolution=None,
        dataset_name=None,
        channel_names=None,
        return_metadata=False,
        dtype=torch.float32,
        scale_factor=1.0
    ):
        self.data_dir = data_dir
        self.n_frames = n_frames
        self.dataset_name = dataset_name
        self.channel_names = channel_names
        self.return_metadata = return_metadata
        self.dtype = dtype
        self.h5_files = [f for f in os.listdir(data_dir) if f.endswith('.h5') or f.endswith('.hdf5')]
        self.h5_files.sort()
        self.data_resolution = data_resolution
        self.labels = np.arange(len(self.h5_files)) # 0, 1.. n-1
        self.scale_factor = scale_factor
        self.dataset_size = self.calculate_size()
    
    def calculate_size(self):
        """
        Calculate the total size of the dataset.
        """
        total_bytes = 0
        for idx in self.labels:
            h5_path = os.path.join(self.data_dir, self.h5_files[idx])
            try:
                with h5py.File(h5_path, 'r') as f:
                    # Get dataset size
                    if 'data' in f:
                        data = f['data']
                        # Calculate memory footprint: shape * item size
                        bytes_per_element = data.dtype.itemsize
                        total_elements = np.prod(data.shape)
                        file_bytes = bytes_per_element * total_elements
                        total_bytes += file_bytes
            except Exception as e:
                print(f"Error reading file {h5_path}: {e}")
        
        return total_bytes / (1024**3)  # Convert bytes to GB

    def select(self,num_samples,seed=42):
        # subsample validation to speed things up
        total_size = len(self.h5_files)
        if num_samples > total_size:
            print(f"All samples are selected !!! {num_samples} / {total_size}")
            return
        else:
            print(f"Sub sampled data !!! {num_samples} / {total_size}")
        seed = 42
        labels = np.arange(total_size)
        rng = np.random.default_rng(seed)
        rng.shuffle(labels)
        labels = labels[:num_samples]
        self.labels = labels
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        idx = self.labels[idx]
        h5_path = os.path.join(self.data_dir, self.h5_files[idx])
        
        with h5py.File(h5_path, 'r') as f:
            data = f['data']
            total_frames = data.shape[0]
            
            max_start_idx = total_frames - self.n_frames
            padding_length = 0
            if max_start_idx < 0:
                #raise ValueError(f"File {h5_path} has fewer frames than requested: {total_frames} vs { self.n_frames}")
                # pad 0 to the beginning
                padding_length = self.n_frames - total_frames
                #raise ValueError(f"File {h5_path} has fewer frames than requested: {total_frames} vs { self.n_frames}/{self.dataset_name}")
                data = np.array(data)
                data = np.pad(data, ((self.n_frames - total_frames,0), (0, 0), (0, 0), (0, 0)), mode='constant')
                assert np.all(data[0] == 0)
                max_start_idx = 0
            start_idx = random.randint(0, max_start_idx)
            frames = data[start_idx:start_idx + self.n_frames]
            assert frames.shape[0] == self.n_frames, f"frames shape {frames.shape} does not match n_frames {self.n_frames}"
            frames = frames.transpose(3, 0, 1, 2)
            frames = torch.from_numpy(frames).to(self.dtype)
            if self.data_resolution is not None:
                frames = resize_input(frames, self.data_resolution)
            
            if not self.return_metadata:
                return frames * self.scale_factor
            else:
                return {
                    'frames': frames * self.scale_factor,
                    'channel_names': self.channel_names,
                    'dataset_name': self.dataset_name,
                    "padding_length":padding_length,
                    # "padded":
                }

class DeterministicHDF5Dataset(RandomHDF5Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._precompute_length()
        
    def _precompute_length(self):
        """precompute number of chunks across every file"""
        self.total_chunks = 0
        self.file_chunks = []
        
        for fname in self.h5_files:
            with h5py.File(os.path.join(self.data_dir, fname), 'r') as f:
                total_frames = f['data'].shape[0]
                chunks = total_frames // self.n_frames
                self.file_chunks.append(chunks)
                self.total_chunks += chunks
                
    def __len__(self):
        return self.total_chunks
        
    def __getitem__(self, idx):
        current = 0
        for file_idx, chunks in enumerate(self.file_chunks):
            if idx < current + chunks:
                chunk_idx = idx - current
                break
            current += chunks
        else:
            raise IndexError("Index out of range")
            
        h5_path = os.path.join(self.data_dir, self.h5_files[file_idx])
        
        with h5py.File(h5_path, 'r') as f:
            data = f['data']
            start_idx = chunk_idx * self.n_frames
            frames = data[start_idx:start_idx + self.n_frames]
            frames = frames.transpose(3, 0, 1, 2)
            frames = torch.from_numpy(frames).to(self.dtype)
            if self.data_resolution is not None:
                frames = resize_input(frames, self.data_resolution)
            return frames

def create_dataloader(data_dir, n_frames, batch_size=32, num_workers=4, 
                     dtype=torch.float32, data_resolution=None, deterministic=False):
    dataset_cls = DeterministicHDF5Dataset if deterministic else RandomHDF5Dataset
    dataset = dataset_cls(data_dir, n_frames, data_resolution=data_resolution, dtype=dtype)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=not deterministic,  # Only shuffle for non-deterministic mode
        num_workers=num_workers,
        pin_memory=True,
    )
