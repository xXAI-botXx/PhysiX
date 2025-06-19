import torch
from typing import List, Dict, Any
from torch.utils.data import Dataset
from megatron.core.datasets.gpt_dataset import GPTDatasetConfig, _PAD_TOKEN_ID, _get_ltor_masks_and_position_ids
from megatron.core.datasets.indexed_dataset import IndexedDataset


class SequenceDataset(Dataset):
    def __init__(
        self,
        dataset_name: str,
        latent_shapes: List[int],
        path_prefix: str,
        config: GPTDatasetConfig,
    ):
        self.dataset_name = dataset_name
        self.latent_shapes = latent_shapes
        self.path_prefix = path_prefix
        self.config = config
        
        self.dataset = IndexedDataset(path_prefix, mmap=True)
    
        self.masks_and_position_ids_are_cacheable = not any(
            [
                self.config.reset_position_ids,
                self.config.reset_attention_mask,
                self.config.eod_mask_loss,
            ]
        )
        self.masks_and_position_ids_are_cached = False
        self.cached_attention_mask = None
        self.cached_loss_mask = None
        self.cached_position_ids = None

        self._pad_token_id = _PAD_TOKEN_ID
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        text = self.dataset[index]
        text = torch.from_numpy(text).long()
        if self.config.add_extra_token_to_sequence:
            tokens = text[:-1].contiguous()
            labels = text[1:].contiguous()
        else:
            tokens = text
            labels = torch.roll(text, shifts=-1, dims=0)
            labels[-1] = self._pad_token_id

        if (
            not self.masks_and_position_ids_are_cacheable
            or not self.masks_and_position_ids_are_cached
        ):
            attention_mask, loss_mask, position_ids = _get_ltor_masks_and_position_ids(
                tokens,
                self.config.tokenizer.eod,
                self.config.reset_position_ids,
                self.config.reset_attention_mask,
                self.config.eod_mask_loss,
                self.config.create_attention_mask,
            )
            if self.masks_and_position_ids_are_cacheable:
                self.cached_attention_mask = attention_mask
                self.cached_loss_mask = loss_mask
                self.cached_position_ids = position_ids
                self.masks_and_position_ids_are_cached = True
        else:
            attention_mask = self.cached_attention_mask
            loss_mask = self.cached_loss_mask
            position_ids = self.cached_position_ids

        # For padded sequences, mask the loss
        loss_mask[labels == self._pad_token_id] = 0.0

        # For padded sequences, ensure the embedding layer can map the token ID
        tokens[tokens == self._pad_token_id] = 0
        labels[labels == self._pad_token_id] = 0

        if self.config.create_attention_mask:
            return {
                "tokens": tokens,
                "labels": labels,
                "attention_mask": attention_mask,
                "loss_mask": loss_mask,
                "position_ids": position_ids,
                "latent_shapes": torch.tensor(self.latent_shapes, dtype=torch.long),
            }
        else:
            return {
                "tokens": tokens,
                "labels": labels,
                "loss_mask": loss_mask,
                "position_ids": position_ids,
                "latent_shapes": torch.tensor(self.latent_shapes, dtype=torch.long),
            }