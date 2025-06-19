import os
import random
import json
import torch
from typing import Optional
from torch.utils.data import DataLoader, Dataset
from lightning import LightningDataModule
from lightning.pytorch.utilities.combined_loader import CombinedLoader
from cosmos1.models.autoregressive.tokenizer.training.data_loader import RandomHDF5Dataset


class MultiDatasetDataModule(LightningDataModule):
    def __init__(
        self,
        metadata_dict,
        n_frames=33,
        batch_size=1,
        num_workers=0,
        pin_memory=False,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)
        
        # Load normalization
        norm_dict = {dataset: {} for dataset in metadata_dict.keys()}
        for dataset, metadata in metadata_dict.items():
            with open(os.path.join(metadata['root_dir'], 'normalization_stats.json'), 'r') as f:
                stats = json.load(f)
            norm_dict[dataset]['means'] = torch.tensor([stat['mean'] for stat in stats])
            norm_dict[dataset]['stds'] = torch.tensor([stat['std'] for stat in stats])
        self.norm_dict = norm_dict

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None):
        # Load datasets only if they're not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            # Create individual datasets
            train_datasets = []
            val_datasets = []
            test_datasets = []
            dataset_names = []
            
            for dataset_name, metadata in self.hparams.metadata_dict.items():
                dataset_names.append(dataset_name)
                
                # Training dataset
                train_datasets.append(
                    RandomHDF5Dataset(
                        data_dir=os.path.join(metadata['root_dir'], 'train'),
                        n_frames=self.hparams.n_frames,
                        dataset_name=dataset_name,
                        data_resolution=metadata['data_resolution'],
                        channel_names=metadata['channel_names'],
                        return_metadata=True,
                    )
                )
                
                # Validation dataset
                val_datasets.append(
                        RandomHDF5Dataset(
                        data_dir=os.path.join(metadata['root_dir'], 'valid'),
                        n_frames=self.hparams.n_frames,
                        dataset_name=dataset_name,
                        data_resolution=metadata['data_resolution'],
                        channel_names=metadata['channel_names'],
                        return_metadata=True,
                    )
                )
                
                # Test dataset
                test_datasets.append(
                        RandomHDF5Dataset(
                        data_dir=os.path.join(metadata['root_dir'], 'test'),
                        n_frames=self.hparams.n_frames,
                        dataset_name=dataset_name,
                        data_resolution=metadata['data_resolution'],
                        channel_names=metadata['channel_names'],
                        return_metadata=True,
                    )
                )
            
            # Create combined datasets
            self.data_train = train_datasets
            self.data_val = val_datasets
            self.data_test = test_datasets
            self.dataset_names = dataset_names

    def denormalize(self, x, dataset_name, valid_indices=None):
        # x: (batch_size, n_channels, n_frames, height, width)
        mean = self.norm_dict[dataset_name]['means'][None, :, None, None, None].to(device=x.device, dtype=x.dtype)
        std = self.norm_dict[dataset_name]['stds'][None, :, None, None, None].to(device=x.device, dtype=x.dtype)
        if valid_indices is not None:
            mean = mean[:, valid_indices, :, :, :]
            std = std[:, valid_indices, :, :, :]
        return x * std + mean

    def _collate_fn(self, batch):
        """
        Custom collate function to handle batches from different datasets.
        """
        frames = torch.stack([item['frames'] for item in batch])
        channel_names = batch[0]['channel_names']
        dataset_name = batch[0]['dataset_name']
        padding_length = [item['padding_length'] for item in batch]
        return frames, dataset_name, channel_names,padding_length

    def train_dataloader(self):
        # Create a list of dataloaders, one for each dataset
        dataloaders = []
        for i, dataset in enumerate(self.data_train):
            dataloaders.append(
                DataLoader(
                    dataset,
                    batch_size=self.hparams.batch_size,
                    shuffle=True,
                    drop_last=False,
                    num_workers=self.hparams.num_workers,
                    pin_memory=self.hparams.pin_memory,
                    collate_fn=self._collate_fn,
                )
            )
        
        # Use RandomIterator to randomly select which dataloader to use for each batch
        # return RandomDatasetIterator(dataloaders)
        return CombinedLoader(dataloaders, mode="max_size_cycle")

    def val_dataloader(self):
        if self.data_val is not None:
            dataloaders = []
            for i, dataset in enumerate(self.data_val):
                dataloaders.append(
                    DataLoader(
                        dataset,
                        batch_size=self.hparams.batch_size,
                        shuffle=False,
                        drop_last=False,
                        num_workers=self.hparams.num_workers,
                        pin_memory=self.hparams.pin_memory,
                        collate_fn=self._collate_fn,
                    )
                )
            
            return dataloaders

    def test_dataloader(self):
        if self.data_test is not None:
            dataloaders = []
            for i, dataset in enumerate(self.data_test):
                dataloaders.append(
                    DataLoader(
                        dataset,
                        batch_size=self.hparams.batch_size,
                        shuffle=False,
                        drop_last=False,
                        num_workers=self.hparams.num_workers,
                        pin_memory=self.hparams.pin_memory,
                        collate_fn=self._collate_fn,
                    )
                )
            
            return dataloaders


# class RandomDatasetIterator:
#     """
#     An iterator that randomly selects batches from different dataloaders.
#     """
#     def __init__(self, dataloaders):
#         self.dataloaders = dataloaders
#         self.iterators = [iter(dl) for dl in dataloaders]
#         self.dataset_weights = [len(dl.dataset) for dl in dataloaders]
#         print('Dataset weights:', self.dataset_weights)
#         total = sum(self.dataset_weights)
#         self.dataset_weights = [w / total for w in self.dataset_weights]
        
#     def __iter__(self):
#         return self
    
#     def __next__(self):
#         # Randomly select a dataset based on its size
#         dataset_idx = random.choices(
#             range(len(self.dataloaders)), 
#             weights=self.dataset_weights, 
#             k=1
#         )[0]
        
#         try:
#             # Try to get the next batch from the selected dataset
#             batch = next(self.iterators[dataset_idx])
#             return batch
#         except StopIteration:
#             # If this iterator is exhausted, create a new one
#             self.iterators[dataset_idx] = iter(self.dataloaders[dataset_idx])
#             batch = next(self.iterators[dataset_idx])
#             return batch
    
#     def __len__(self):
#         # The length is the sum of all dataloader lengths
#         return sum(len(dl) for dl in self.dataloaders)


# class SequentialDatasetIterator:
#     """
#     An iterator that processes datasets sequentially - completes one dataset 
#     before moving to the next. Ideal for validation and testing.
#     """
#     def __init__(self, dataloaders):
#         self.dataloaders = dataloaders
#         self.current_dataloader_idx = 0
#         self.current_iterator = iter(self.dataloaders[0]) if self.dataloaders else None
        
#     def __iter__(self):
#         # Reset to the first dataloader
#         self.current_dataloader_idx = 0
#         if self.dataloaders:
#             self.current_iterator = iter(self.dataloaders[0])
#         return self
    
#     def __next__(self):
#         if not self.dataloaders:
#             raise StopIteration
            
#         try:
#             # Try to get the next batch from the current dataset
#             return next(self.current_iterator)
#         except StopIteration:
#             # Current dataset is exhausted, move to the next one
#             self.current_dataloader_idx += 1
            
#             # If we've gone through all datasets, we're done
#             if self.current_dataloader_idx >= len(self.dataloaders):
#                 raise StopIteration
                
#             # Move to the next dataset
#             self.current_iterator = iter(self.dataloaders[self.current_dataloader_idx])
#             return next(self.current_iterator)
    
#     def __len__(self):
#         # The length is the sum of all dataloader lengths
#         return sum(len(dl) for dl in self.dataloaders)


# # Example of how to use the modified DataModule
# metadata_dict = {
#   "active_matter": {
#     "root_dir": "/eagle/MDClimSim/tungnd/data/the_well/normalized/active_matter",
#     "data_resolution": [256, 256],
#     "channel_names": ["concentration", "velocity_x", "velocity_y", "D_xx", "D_xy", "D_yx", "D_yy", "E_xx", "E_xy", "E_yx", "E_yy"],
#   },
#   "shear_flow_4c": {
#     "root_dir": "/eagle/MDClimSim/tungnd/data/the_well/normalized/shear_flow_4c",
#     "data_resolution": [512, 256],
#     "channel_names": ["tracer", "pressure", "velocity_x", "velocity_y"],
#   },
# }

# data_module = MultiDatasetDataModule(
#     metadata_dict=metadata_dict,
#     n_frames=33,
#     batch_size=2,
#     num_workers=1,
# )
# data_module.setup()
# # dataloader = data_module.train_dataloader()
# dataloader = data_module.val_dataloader()
# for i, (frames, dataset_name, channel_names) in enumerate(dataloader):
#     print(f"Batch {i}: {frames.shape}, {dataset_name}, {channel_names}")
#     if i == 10:
#         break
