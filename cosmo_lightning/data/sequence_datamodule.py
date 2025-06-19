import os
import numpy as np
from typing import Optional, List
from torch.utils.data.dataloader import default_collate
from torch.utils.data import DataLoader, Dataset
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


class SequenceDatamodule(LightningDataModule):
    def __init__(
        self,
        dataset_name: str,
        root_dir: str,
        latent_shapes: List[int],
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
        num_workers: int = 4,
        pin_memory: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        
        self.seq_length = np.prod(latent_shapes)
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
            dataloader_type="cyclic"
        )
    
    def setup(self, stage: Optional[str] = None):
        # Load datasets only if they're not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            self.data_train = SequenceDataset(
                dataset_name=self.hparams.dataset_name,
                latent_shapes=self.hparams.latent_shapes,
                path_prefix=os.path.join(self.hparams.root_dir, "train", "embeddings"),
                config=self.gpt_dataset_config,
            )
            
            self.data_val = SequenceDataset(
                dataset_name=self.hparams.dataset_name,
                latent_shapes=self.hparams.latent_shapes,
                path_prefix=os.path.join(self.hparams.root_dir, "valid", "embeddings"),
                config=self.gpt_dataset_config,
            )
            
            self.data_test = SequenceDataset(
                dataset_name=self.hparams.dataset_name,
                latent_shapes=self.hparams.latent_shapes,
                path_prefix=os.path.join(self.hparams.root_dir, "test", "embeddings"),
                config=self.gpt_dataset_config,
            )
    
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
        return self._create_dataloader(self.data_val, mode="validation")

    def test_dataloader(self):
        return self._create_dataloader(self.data_test, mode="test")
    
    @property
    def gpt_dataset_config(self) -> "GPTDatasetConfig":
        build_kwargs = {}
        build_kwargs["blend_per_split"] = [
            get_blend_from_list([os.path.join(self.hparams.root_dir, "train", "embeddings")]),
            get_blend_from_list([os.path.join(self.hparams.root_dir, "valid", "embeddings")]),
            get_blend_from_list([os.path.join(self.hparams.root_dir, "test", "embeddings")]),
        ]

        return GPTDatasetConfig(
            random_seed=self.seed,
            sequence_length=self.seq_length,
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