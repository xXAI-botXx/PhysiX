import os
import random
import json
import torch
from collections import defaultdict
from typing import Optional, Dict, List
from torch.utils.data import DataLoader, Dataset, IterableDataset
from lightning import LightningDataModule
from lightning.pytorch.utilities.rank_zero import rank_zero_info
from cosmos1.models.autoregressive.tokenizer.training.data_loader import RandomHDF5Dataset


class ContinuousMultiDatasetSampler(IterableDataset):
    def __init__(self, datasets, seed=None):
        """
        Create a continuous sampler that cycles through multiple datasets.
        
        Args:
            datasets: List of datasets to sample from
            seed: Random seed for reproducibility
        """
        super().__init__()
        self.datasets = datasets
        self.seed = seed if seed is not None else random.randint(0, 2**32 - 1)
        self.dataset_lengths = [len(dataset) for dataset in datasets]
        self.rng = random.Random(self.seed)
        
        # Weights for random selection based on dataset sizes
        total_samples = sum(self.dataset_lengths)
        self.weights = [length / total_samples for length in self.dataset_lengths]
    
    def __iter__(self):
        # Create samplers for each dataset
        iterators = [iter(torch.utils.data.RandomSampler(
            dataset, 
            replacement=True,  # Sample with replacement for continuous iteration
            num_samples=2**62  # A very large number
        )) for dataset in self.datasets]
        
        while True:
            # Select a dataset based on its relative size
            dataset_idx = self.rng.choices(range(len(self.datasets)), weights=self.weights, k=1)[0]
            
            # Get the next index from the selected dataset's sampler
            try:
                sample_idx = next(iterators[dataset_idx])
            except StopIteration:
                # Reset the iterator if it's exhausted (shouldn't happen with replacement=True)
                iterators[dataset_idx] = iter(torch.utils.data.RandomSampler(
                    self.datasets[dataset_idx], replacement=True, num_samples=2**62
                ))
                sample_idx = next(iterators[dataset_idx])
            
            # Return the sample along with metadata about which dataset it came from
            yield {
                'data': self.datasets[dataset_idx][sample_idx],
                'dataset_idx': dataset_idx
            }


class DistributedMultiDatasetDataModule(LightningDataModule):
    """A Lightning DataModule that distributes datasets across GPUs in a balanced way."""
    def __init__(
        self,
        metadata_dict: Dict,
        n_frames: int = 33,
        batch_size: int = 1,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        
        # Load normalization stats
        norm_dict = {dataset: {} for dataset in metadata_dict.keys()}
        for dataset, metadata in metadata_dict.items():
            with open(os.path.join(metadata['root_dir'], 'normalization_stats.json'), 'r') as f:
                stats = json.load(f)
            norm_dict[dataset]['means'] = torch.tensor([stat['mean'] for stat in stats])
            norm_dict[dataset]['stds'] = torch.tensor([stat['std'] for stat in stats])
        self.norm_dict = norm_dict
        
        self.data_train: Optional[Dict[str, Dataset]] = None
        self.data_val: Optional[Dict[str, Dataset]] = None
        self.data_test: Optional[Dict[str, Dataset]] = None
        
        # Will be set during setup
        self.dataset_names = list(metadata_dict.keys())
        self.dataset_assignment = None
        self.is_distributed = False
        self.world_size = 1
        self.global_rank = 0
        
    def setup(self, stage: Optional[str] = None):
        """Set up datasets and determine dataset assignment based on distributed environment."""
        # Load datasets only if they're not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            # Create dictionaries to hold datasets
            train_datasets = {}
            val_datasets = []
            test_datasets = []
            
            # Check if we're in a distributed environment
            try:
                self.world_size = torch.distributed.get_world_size()
                self.global_rank = torch.distributed.get_rank()
                self.is_distributed = True
                rank_zero_info(f"Running in distributed mode with world_size={self.world_size}")
            except (RuntimeError, ValueError, ImportError):
                self.world_size = 1
                self.global_rank = 0
                self.is_distributed = False
                rank_zero_info("Running in non-distributed mode")
            
            # Create datasets for each dataset type
            for dataset_name, metadata in self.hparams.metadata_dict.items():                
                # Training dataset
                train_datasets[dataset_name] = RandomHDF5Dataset(
                    data_dir=os.path.join(metadata['root_dir'], 'train'),
                    n_frames=self.hparams.n_frames,
                    dataset_name=dataset_name,
                    data_resolution=metadata['data_resolution'],
                    channel_names=metadata['channel_names'],
                    return_metadata=True,
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
                
            # Store datasets
            self.data_train = train_datasets
            self.data_val = val_datasets
            self.data_test = test_datasets
            
            # Determine dataset assignments for distributed training
            self.dataset_assignment = self._assign_datasets_to_ranks()
            
            # Log dataset assignment (only on rank 0)
            if self.global_rank == 0:
                rank_zero_info(f"Dataset assignment: {self.dataset_assignment}")
    
    def _assign_datasets_to_ranks(self) -> Dict[str, List[int]]:
        """
        Assign datasets to ranks in a balanced way.
        Returns a dictionary mapping dataset names to lists of ranks.
        
        If num_datasets <= world_size:
            - Each dataset gets at least world_size // num_datasets GPUs
            - Remaining GPUs are distributed to datasets in order
        
        If num_datasets > world_size:
            - Each GPU gets at least one dataset
            - Some GPUs may get multiple datasets
        """
        num_datasets = len(self.dataset_names)
        assignments = defaultdict(list)
        
        if not self.is_distributed or self.world_size == 1:
            # If not distributed, all datasets are processed by rank 0
            for dataset_name in self.dataset_names:
                assignments[dataset_name] = [0]
            return dict(assignments)
        
        if num_datasets <= self.world_size:
            # Case 1: More GPUs than datasets
            # Calculate base GPUs per dataset
            gpus_per_dataset = self.world_size // num_datasets
            remainder = self.world_size % num_datasets
            
            # Sort datasets by estimated size (using length as proxy)
            # This ensures larger datasets get more GPUs if there are remainders
            dataset_sizes = [(name, self.data_train[name].dataset_size) for name in self.dataset_names]
            dataset_sizes.sort(key=lambda x: x[1], reverse=True)
            
            # Assign ranks to datasets
            rank_idx = 0
            for dataset_name, _ in dataset_sizes:
                # Assign base number of GPUs
                for _ in range(gpus_per_dataset):
                    assignments[dataset_name].append(rank_idx)
                    rank_idx += 1
                
                # Assign one extra GPU if there are remainders
                if remainder > 0:
                    assignments[dataset_name].append(rank_idx)
                    rank_idx += 1
                    remainder -= 1
        else:
            # Case 2: More datasets than GPUs
            # This is not your primary use case, but handle it anyway by
            # assigning multiple datasets to each GPU in a balanced way
            datasets_per_gpu = num_datasets // self.world_size
            remainder = num_datasets % self.world_size
            
            dataset_idx = 0
            for rank in range(self.world_size):
                # Assign base number of datasets to this rank
                for _ in range(datasets_per_gpu):
                    if dataset_idx < num_datasets:
                        dataset_name = self.dataset_names[dataset_idx]
                        assignments[dataset_name].append(rank)
                        dataset_idx += 1
                
                # Assign one extra dataset if there are remainders
                if remainder > 0 and dataset_idx < num_datasets:
                    dataset_name = self.dataset_names[dataset_idx]
                    assignments[dataset_name].append(rank)
                    dataset_idx += 1
                    remainder -= 1
        
        return dict(assignments)
    
    def denormalize(self, x, dataset_name, valid_indices=None):
        """Denormalize data using stored mean and std values."""
        mean = self.norm_dict[dataset_name]['means'][None, :, None, None, None].to(device=x.device, dtype=x.dtype)
        std = self.norm_dict[dataset_name]['stds'][None, :, None, None, None].to(device=x.device, dtype=x.dtype)
        if valid_indices is not None:
            mean = mean[:, valid_indices, :, :, :]
            std = std[:, valid_indices, :, :, :]
        return x * std + mean
    
    def _collate_fn(self, batch):
        """Custom collate function to handle batches."""
        frames = torch.stack([item['frames'] for item in batch])
        channel_names = batch[0]['channel_names']
        dataset_name = batch[0]['dataset_name']
        padding_length = [item['padding_length'] for item in batch]
        return frames, dataset_name, channel_names, padding_length
    
    def _collate_continuous_batch(self, batch):
        """Custom collate function for continuous batch sampling."""
        # Unpack the data and dataset indices
        data_items = [item['data'] for item in batch]
        dataset_indices = [item['dataset_idx'] for item in batch]
        
        # Collate based on the original format
        if isinstance(data_items[0], dict):
            # If the dataset returns dictionaries
            frames = torch.stack([item['frames'] for item in data_items])
            channel_names = data_items[0]['channel_names']
            dataset_name = data_items[0]['dataset_name']
            padding_length = [item['padding_length'] for item in data_items]
            return frames, dataset_name, channel_names, padding_length
        else:
            # If the dataset returns tensors directly
            return torch.stack(data_items)
    
    def _get_my_datasets(self, split_datasets):
        """Get dataset portions that should be processed by the current rank."""
        if not self.is_distributed or self.world_size == 1:
            # If not distributed, use all datasets
            return list(split_datasets.values())
        
        # Otherwise, use only dataset portions assigned to this rank
        my_datasets = []
        for dataset_name, dataset in split_datasets.items():
            if self.global_rank in self.dataset_assignment[dataset_name]:
                # Calculate this rank's portion of the dataset
                assigned_ranks = self.dataset_assignment[dataset_name]
                rank_position = assigned_ranks.index(self.global_rank)
                total_assigned_ranks = len(assigned_ranks)
                
                # Determine indices for this rank's portion
                total_samples = len(dataset)
                samples_per_rank = total_samples // total_assigned_ranks
                start_idx = rank_position * samples_per_rank
                end_idx = start_idx + samples_per_rank if rank_position < total_assigned_ranks - 1 else total_samples
                
                # Create subset indices
                indices = list(range(start_idx, end_idx))
                
                # Create a subset of the dataset using torch.utils.data.Subset
                subset = torch.utils.data.Subset(dataset, indices)
                
                my_datasets.append(subset)
        
        return my_datasets
    
    def train_dataloader(self):
        """Return training dataloaders for datasets assigned to this rank."""
        my_datasets = self._get_my_datasets(self.data_train)
        
        # If no datasets assigned to this rank, return an empty dataloader
        if not my_datasets:
            # Create a tiny dummy dataset to avoid errors
            dummy_dataset = torch.utils.data.TensorDataset(torch.zeros(1, 1))
            return DataLoader(dummy_dataset, batch_size=1)
        
        # Create a continuous sampler from all assigned datasets
        continuous_dataset = ContinuousMultiDatasetSampler(my_datasets)
        
        # Create a single dataloader
        return DataLoader(
            continuous_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self._collate_continuous_batch
        )
    
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
