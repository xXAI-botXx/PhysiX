import os
import json
import torch
from typing import Optional
from torch.utils.data import DataLoader, Dataset
from lightning import LightningDataModule
from cosmos1.models.autoregressive.tokenizer.training.data_loader import RandomHDF5Dataset


class VAEDataModule(LightningDataModule):
    def __init__(
        self,
        root_dir,
        data_resolution,
        channel_names,
        n_frames=33,
        batch_size=1,
        num_workers=0,
        pin_memory=False,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)
        
        # load normalization
        with open(os.path.join(root_dir, 'normalization_stats.json'), 'r') as f:
            stats = json.load(f)
        self.norm_means = torch.tensor([stat['mean'] for stat in stats])
        self.norm_stds = torch.tensor([stat['std'] for stat in stats])        

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    # def denormalize(self, x):
    #     # x: (batch_size, n_frames, n_channels, height, width)
    #     device, dtype = x.device, x.dtype
    #     x = x * self.norm_stds[None, :, None, None, None].to(device=device, dtype=dtype) + self.norm_means[None, :, None, None, None].to(device=device, dtype=dtype)
    #     return x

    def setup(self, stage: Optional[str] = None):
        # load datasets only if they're not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            self.data_train = RandomHDF5Dataset(
                data_dir=os.path.join(self.hparams.root_dir, 'train'),
                n_frames=self.hparams.n_frames,
                data_resolution=self.hparams.data_resolution,
            )
            
            self.data_val = RandomHDF5Dataset(
                data_dir=os.path.join(self.hparams.root_dir, 'valid'),
                n_frames=self.hparams.n_frames,
                data_resolution=self.hparams.data_resolution,
            )

            self.data_test = RandomHDF5Dataset(
                data_dir=os.path.join(self.hparams.root_dir, 'test'),
                n_frames=self.hparams.n_frames,
                data_resolution=self.hparams.data_resolution,
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

    def val_dataloader(self):
        if self.data_val is not None:
            return DataLoader(
                self.data_val,
                batch_size=self.hparams.batch_size,
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

# datamodule = LatentVideoDataModule(
#     latent_root_dir='/eagle/MDClimSim/tungnd/data/wb2/1.40625deg_latent_16_down/',
#     raw_root_dir='/eagle/MDClimSim/tungnd/data/wb2/1.40625deg_from_full_res_1_step_6hr_h5df',
#     variables=[
#         "2m_temperature",
#         "10m_u_component_of_wind",
#         "10m_v_component_of_wind",
#         "geopotential_500",
#         "temperature_850"
#     ],
#     return_raw=True,
#     steps=17,
#     interval=6,
#     data_freq=6,
#     batch_size=8,
#     val_batch_size=8,
#     num_workers=1,
#     pin_memory=False
# )
# datamodule.setup()
# for batch in datamodule.train_dataloader():
#     latent_video, raw_video = batch
#     print (latent_video.shape, raw_video.shape)
#     break