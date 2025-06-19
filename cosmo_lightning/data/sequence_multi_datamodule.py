import os
import numpy as np
import torch
from typing import Optional, List, Dict
from torch.utils.data.dataloader import default_collate
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from lightning import LightningDataModule
from megatron.core.datasets.gpt_dataset import GPTDatasetConfig
from megatron.core.datasets.utils import get_blend_from_list
from nemo.lightning.data import WrappedDataLoader
from cosmo_lightning.data.custom_data_sampler import CustomMegatronDataSampler
from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
from nemo.utils.import_utils import safe_import
from cosmo_lightning.data.sequence_dataset import SequenceDataset

_, HAVE_TE = safe_import("transformer_engine")


class ExactSizeDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, target_size):
        self.dataset = dataset
        self.original_size = len(dataset)
        self.target_size = target_size
        
    def __len__(self):
        return self.target_size
    
    def __getitem__(self, idx):
        # Map the index to the original dataset
        return self.dataset[idx % self.original_size]


class SequenceMultiDatamodule(LightningDataModule):
    def __init__(
        self,
        data_metadata: Dict,
        tokenizer: Optional["TokenizerSpec"] = None,
        reset_position_ids: bool = False,
        create_attention_mask: bool = False,
        reset_attention_mask: bool = False,
        eod_mask_loss: bool = False,
        index_mapping_dir: Optional[str] = None,
        num_dataset_builder_threads: int = 1,
        seed: int = 1234,
        global_batch_size: int = 1,
        micro_batch_size: int = 1,
        init_consumed_samples: int = 0,
        init_global_step: int = 0,
        num_workers: int = 4,
        pin_memory: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        
        self.seq_length = np.max([np.prod(data_metadata[dataset]["latent_shapes"]) for dataset in data_metadata])
        self.tokenizer = tokenizer or get_nmt_tokenizer("megatron", "GPT2BPETokenizer")
        self.reset_position_ids = reset_position_ids
        self.create_attention_mask = create_attention_mask or not HAVE_TE
        self.reset_attention_mask = reset_attention_mask
        self.eod_mask_loss = eod_mask_loss
        self.index_mapping_dir = index_mapping_dir
        self.num_dataset_builder_threads = num_dataset_builder_threads
        self.seed = seed
        
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        
        self.data_sampler = CustomMegatronDataSampler(
            seq_len=self.seq_length,
            micro_batch_size=micro_batch_size,
            global_batch_size=global_batch_size,
            rampup_batch_size=None,
            dataloader_type="cyclic",
            init_consumed_samples=init_consumed_samples,
            init_global_step=init_global_step,
        )
    
    def setup(self, stage: Optional[str] = None):
        # Load datasets only if they're not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            all_train_datasets = []
            dataset_sizes = []

            for dataset_name, metadata in self.hparams.data_metadata.items():
                seq_length = np.prod(metadata["latent_shapes"])
                dataset = SequenceDataset(
                    dataset_name=dataset_name,
                    latent_shapes=metadata["latent_shapes"],
                    path_prefix=os.path.join(metadata["root_dir"], "train", "embeddings"),
                    config=self.get_gpt_dataset_config(metadata["root_dir"], seq_length),
                )
                all_train_datasets.append(dataset)
                dataset_sizes.append(len(dataset))
            
            max_size = max(dataset_sizes)
            
            balanced_datasets = []
            for i, dataset in enumerate(all_train_datasets):
                if dataset_sizes[i] < max_size:
                    balanced_datasets.append(ExactSizeDataset(dataset, max_size))
                else:
                    balanced_datasets.append(dataset)
            
            self.data_train = ConcatDataset(balanced_datasets)
            
            all_val_datasets = []
            for dataset_name, metadata in self.hparams.data_metadata.items():
                seq_length = np.prod(metadata["latent_shapes"])
                all_val_datasets.append(
                    SequenceDataset(
                        dataset_name=dataset_name,
                        latent_shapes=metadata["latent_shapes"],
                        path_prefix=os.path.join(metadata["root_dir"], "valid", "embeddings"),
                        config=self.get_gpt_dataset_config(metadata["root_dir"], seq_length),
                    )
                )
            # self.data_val = ConcatDataset(all_val_datasets)
            self.data_val = all_val_datasets
            
            all_test_datasets = []
            for dataset_name, metadata in self.hparams.data_metadata.items():
                seq_length = np.prod(metadata["latent_shapes"])
                all_test_datasets.append(
                    SequenceDataset(
                        dataset_name=dataset_name,
                        latent_shapes=metadata["latent_shapes"],
                        path_prefix=os.path.join(metadata["root_dir"], "test", "embeddings"),
                        config=self.get_gpt_dataset_config(metadata["root_dir"], seq_length),
                    )
                )
            # self.data_test = ConcatDataset(all_test_datasets)
            self.data_test = all_test_datasets
    
    def _create_dataloader(self, dataset, mode, **kwargs) -> WrappedDataLoader:
        self.init_global_step = self.trainer.global_step
        self.data_sampler.init_global_step = self.init_global_step
        dataloader = WrappedDataLoader(
            mode=mode,
            dataset=dataset,
            # batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=getattr(dataset, "collate_fn", default_collate),
            **kwargs,
        )
        return dataloader
    
    def train_dataloader(self):
        return self._create_dataloader(self.data_train, mode="train", shuffle=True)

    def val_dataloader(self):
        # return self._create_dataloader(self.data_val, mode="validation")
        dataloaders = []
        for i, dataset in enumerate(self.data_val):
            dataloaders.append(
                WrappedDataLoader(
                    mode="validation",
                    dataset=dataset,
                    num_workers=self.hparams.num_workers,
                    pin_memory=self.hparams.pin_memory,
                    collate_fn=getattr(dataset, "collate_fn", default_collate),
                )
            )
        
        return dataloaders

    def test_dataloader(self):
        # return self._create_dataloader(self.data_test, mode="test")
        dataloaders = []
        for i, dataset in enumerate(self.data_test):
            dataloaders.append(
                WrappedDataLoader(
                    mode="test",
                    dataset=dataset,
                    num_workers=self.hparams.num_workers,
                    pin_memory=self.hparams.pin_memory,
                    collate_fn=getattr(dataset, "collate_fn", default_collate),
                )
            )
        return dataloaders
    
    def get_gpt_dataset_config(self, root_dir, seq_length) -> GPTDatasetConfig:
        build_kwargs = {}
        build_kwargs["blend_per_split"] = [
            get_blend_from_list([os.path.join(root_dir, "train", "embeddings")]),
            get_blend_from_list([os.path.join(root_dir, "valid", "embeddings")]),
            get_blend_from_list([os.path.join(root_dir, "test", "embeddings")]),
        ]

        return GPTDatasetConfig(
            random_seed=self.seed,
            sequence_length=seq_length,
            add_extra_token_to_sequence=False,
            tokenizer=self.tokenizer,
            path_to_cache=self.index_mapping_dir,
            reset_position_ids=self.reset_position_ids,
            create_attention_mask=self.create_attention_mask,
            reset_attention_mask=self.reset_attention_mask,
            eod_mask_loss=self.eod_mask_loss,
            num_dataset_builder_threads=self.num_dataset_builder_threads,
            **build_kwargs,
        )