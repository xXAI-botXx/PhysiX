import os
import h5py
import torch
from glob import glob
from typing import Optional
from torch.utils.data import DataLoader, Dataset
from lightning import LightningDataModule
from cosmos1.models.autoregressive.nemo.utils import resize_input


class VideoDataset(Dataset):
    def __init__(
        self,
        root_dir,
        dimensions,
        num_context_frames=13,
    ):
        super().__init__()
        self.file_paths = sorted(glob(os.path.join(root_dir, "*.hdf5")))
        self.dimensions = dimensions
        self.num_context_frames = num_context_frames
        
        # Pre-scan to determine video clips in each file without loading full data
        self.clip_indices = []
        
        for file_idx, file_path in enumerate(self.file_paths):
            with h5py.File(file_path, "r") as f:
                n_frames = f['data'].shape[0]
            
            if n_frames >= num_context_frames:
                # For each possible sliding window in the file
                for start_frame in range(n_frames - num_context_frames + 1):
                    self.clip_indices.append((file_idx, start_frame))
            else:
                # If file is too short, it's just one clip with padding
                self.clip_indices.append((file_idx, 0))
    
    def __len__(self):
        return len(self.clip_indices)
    
    def __getitem__(self, idx):
        file_idx, start_frame = self.clip_indices[idx]
        
        with h5py.File(self.file_paths[file_idx], "r") as f:
            n_frames = f['data'].shape[0]
            
            if n_frames >= self.num_context_frames:
                # Only load the frames needed for this clip
                frame_data = f['data'][start_frame:start_frame + self.num_context_frames][:]
                video = torch.from_numpy(frame_data).float()
                video = video.permute(3, 0, 1, 2)  # [C, T, H, W]
                video = resize_input(video, self.dimensions)
            else:
                # For short videos, load all frames and pad
                frame_data = f['data'][:][:]
                video = torch.from_numpy(frame_data).float()
                # Handle padding
                padding = video[-1].unsqueeze(0).repeat(self.num_context_frames - n_frames, 1, 1, 1)
                video = torch.cat([video, padding], dim=0)
                video = video.permute(3, 0, 1, 2)  # [C, T, H, W]
                video = resize_input(video, self.dimensions)
            
        return video


class GenEmbeddingDataModule(LightningDataModule):
    def __init__(
        self,
        root_dir,
        dimensions,
        batch_size=1,
        num_workers=0,
        pin_memory=False,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None):
        # Load datasets only if they're not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            self.data_train = VideoDataset(
                root_dir=os.path.join(self.hparams.root_dir, "train"),
                dimensions=self.hparams.dimensions,
            )
            self.data_val = VideoDataset(
                root_dir=os.path.join(self.hparams.root_dir, "valid"),
                dimensions=self.hparams.dimensions,
            )
            self.data_test = VideoDataset(
                root_dir=os.path.join(self.hparams.root_dir, "test"),
                dimensions=self.hparams.dimensions,
            )

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )
