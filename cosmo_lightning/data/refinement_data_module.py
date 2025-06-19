import os
import json
import torch
from typing import Optional
from torch.utils.data import DataLoader, Dataset
from lightning import LightningDataModule
from cosmos1.models.autoregressive.tokenizer.training.data_loader import RandomHDF5Dataset
from torch.utils.data import Dataset
import glob
import numpy as np
from einops import rearrange
class RefinementDataset(Dataset):
    def __init__(
        self,
        data_dir,
        context_frames,
        grounding_frames,
        total_frames=13,
        max_start_idx=-1,
        baseline_format=False,
        other_dirs=[],
    ):
        self.data_dir = data_dir
        self.baseline_format = baseline_format
        self.context_frames = context_frames
        self.grounding_frames = grounding_frames
        self.files = glob.glob(os.path.join(data_dir, '*.npz'))
        for other_dir in other_dirs:
            new_files = glob.glob(os.path.join(other_dir, '*.npz'))
            self.files.extend(new_files)
        # data1 = self.load_file(self.files[0])
        self.data = []
        for file in self.files:
            #min_offset_frames = min(grounding_frames)
            target_idx = context_frames
            while target_idx < total_frames:
                if max_start_idx > 0 and target_idx > max_start_idx:
                    break
                payload = dict(
                    filepath=file,
                    target_idx=target_idx
                )
                target_idx += 1
                self.data.append(payload)
        
    def load_file(self, file_path):
        # Implement this method to load the data from the file
        # and return the relevant tensors
        fp = np.load(file_path)
        data = dict(
            input_frames=fp['input_frames'],
            video_decoded=fp['video_decoded'],
            pred_frames=fp['pred_frames'],
            ground_truth=fp['ground_truth']
        )
        # breakpoint()
        fp.close()
        return dict(data)
    
    def __len__(self):
        # Implement this method to return the total number of samples in the dataset
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        payload = self.load_file(item['filepath'])
        start_idx = item['target_idx']
        # breakpoint()
        # diff = np.abs(payload['ground_truth'][:,:,:5] - payload['pred_frames'][None][:,:,:5])
        # diff
        
        if self.baseline_format:
            input_frames = payload['pred_frames'][None].astype(np.float32) # N C T_CONTEXT H W
            input_frames[:,:,:self.context_frames]= payload['ground_truth'][:,:,:self.context_frames]
            input_frames = input_frames[:,:,start_idx-self.context_frames:start_idx] # N C T_CONTEXT H W
        else:
            input_frames = payload['ground_truth'][:,:,:self.context_frames] # N C T_CONTEXT H W
        target_frame = payload['ground_truth'][:,:,start_idx] # N C  H W
        grounding_idx = [start_idx + i for i in self.grounding_frames]
        grounding_frames  = payload['pred_frames'][None][:,:,grounding_idx] # N C T_GROUNDING H W
        input_frames = torch.tensor(input_frames) # N C T_CONTEXT H W
        grounding_frames = torch.tensor(grounding_frames) # N C T_GROUNDING H W
        target_frame = torch.tensor(target_frame) # N C H W
        # breakpoint()
        x = torch.cat([input_frames, grounding_frames], dim=2) # N C T_CONTEXT + T_GROUNDING H W
        # breakpoint()
        if self.baseline_format:
            x = rearrange(x, 'n c t h w -> n (t c) h w')  # N C T_CONTEXT + T_GROUNDING H W
        else:
            x = x.flatten(1,2)  #( N C T_CONTEXT + T_GROUNDING )H W
        assert len(x.shape) ==4  # N C H W
        # print(start_idx)
        return x[0],target_frame[0],start_idx
    
class RefinementDataModule(LightningDataModule):
    def __init__(
        self,
        root_dir,
        context_frames,
        total_frames,
        batch_size=1,
        num_workers=0,
        grounding_frames=None,
        pin_memory=False,
        normalization_stats=None,
        max_start_idx=-1,
        other_dirs=[],
        val_dir='',
        baseline_format=False
        
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)
        
        # load normalization
        
        with open(os.path.join(normalization_stats, 'normalization_stats.json'), 'r') as f:
            stats = json.load(f)
        self.norm_means = torch.tensor([stat['mean'] for stat in stats])
        self.norm_stds = torch.tensor([stat['std'] for stat in stats])        

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        self.val_dir = val_dir

    def setup(self, stage: Optional[str] = None):
        # load datasets only if they're not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            self.data_train = RefinementDataset(
                data_dir=os.path.join(self.hparams.root_dir, 'train'),
                other_dirs=[os.path.join(x, 'train') for x in self.hparams.other_dirs],
                context_frames=self.hparams.context_frames,
                grounding_frames=self.hparams.grounding_frames,
                total_frames=self.hparams.total_frames,
                max_start_idx=self.hparams.max_start_idx,
                baseline_format=self.hparams.baseline_format,
            )
            val_path = self.hparams.root_dir if not self.val_dir else self.val_dir
            self.data_val = RefinementDataset(
                # data_dir=os.path.join(val_path, 'test'),
                data_dir=os.path.join(val_path, 'test'),
                context_frames=self.hparams.context_frames,
                grounding_frames=self.hparams.grounding_frames,
                total_frames=self.hparams.total_frames,
                max_start_idx=self.hparams.max_start_idx,
                baseline_format=self.hparams.baseline_format,
            )
            # breakpoint()

            self.data_test = RefinementDataset(
                # data_dir=os.path.join(val_path, 'test'),
                data_dir=os.path.join(self.hparams.root_dir, 'test'),
                context_frames=self.hparams.context_frames,
                grounding_frames=self.hparams.grounding_frames,
                total_frames=self.hparams.total_frames,
                max_start_idx=self.hparams.max_start_idx,
                baseline_format=self.hparams.baseline_format,
            )

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )
        
    def denormalize(self, x,valid_indices=None):
        # x: (batch_size, n_frames, n_channels, height, width)
        device, dtype = x.device, x.dtype
        # breakpoint()
        # x = x[None]
        mean = self.norm_means[None, :, None, None, None].to(device=x.device, dtype=x.dtype)
        std = self.norm_stds[None, :, None, None, None].to(device=x.device, dtype=x.dtype)
        if valid_indices is not None:
            mean = mean[:, valid_indices, :, :, :]
            std = std[:, valid_indices, :, :, :]
        # breakpoint()
        # squeeze temporal  axis
        return x * std.squeeze(-3) + mean[0].squeeze(-3)

    def val_dataloader(self):
        if self.data_val is not None:
            return DataLoader(
                self.data_val,
                batch_size=1,
                shuffle=False,
                drop_last=False,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
            )

    def test_dataloader(self):
        if self.data_test is not None:
            return DataLoader(
                self.data_test,
                batch_size=self.hparams.batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
            )
            
if __name__ == "__main__":
    dataset = RefinementDataset(
        data_dir='/data0/jacklishufan/cosmos-refinement/vi/train',
        context_frames=5,
        grounding_frames=[-1],
    )