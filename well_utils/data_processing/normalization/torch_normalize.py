import json
from pathlib import Path
from typing import Union, Optional, List

import torch
import torch.nn as nn


class TorchNormalizationApplier(nn.Module):
    """
    Applies normalization and denormalization using precomputed statistics.

    This module is designed to work with Torch tensors and assumes that the input tensor
    is structured as (batch[optional], channels, timestep, height, width). For example:
        - Without batch: (channels, timestep, height, width)
        - With batch: (batch, channels, timestep, height, width)

    The normalization is implemented in a fully Torch–compatible way, so no CPU–GPU
    transfers or numpy conversions are necessary.
    """
    def __init__(
        self,
        stats_path: Union[str, Path],
        normalization_type: str = "standard",
        constant_channels: List[int] = []
    ):
        """
        Initializes the normalization module from a JSON statistics file.

        Args:
            stats_path (Union[str, Path]): Path to the JSON file with channel statistics.
            normalization_type (str): Either "minmax" or "standard". Defaults to "standard".
        """
        super().__init__()
        with open(Path(stats_path), "r") as f:
            stats = json.load(f)

        self.n_channels = len(stats)
        self.normalization_type = normalization_type
        self.constant_channels = constant_channels
        self.n_constant_channels = len(constant_channels) if constant_channels is not None else 0
        eps = 1e-7

        if normalization_type == "minmax":
            # Pre-compute scale and offset for min–max normalization.
            scale_values = [2 / (c["max"] - c["min"] + eps) for c in stats]
            offset_values = [-1 - (2 * c["min"]) / (c["max"] - c["min"] + eps) for c in stats]
        elif normalization_type == "standard":
            # Pre-compute scale and offset for standardization.
            scale_values = [1 / (c["std"] + eps) for c in stats]
            offset_values = [-c["mean"] / (c["std"] + eps) for c in stats]
        else:
            raise ValueError(f"Unsupported normalization type: {normalization_type}")

        self.register_buffer("scale", torch.tensor(scale_values))
        self.register_buffer("offset", torch.tensor(offset_values))
        self.register_buffer("constant_scale", torch.tensor([val for i, val in enumerate(scale_values) if i not in constant_channels]))
        self.register_buffer("constant_offset", torch.tensor([val for i, val in enumerate(offset_values) if i not in constant_channels]))

    def _validate_tensor_shape(self, x: torch.Tensor, ignore_constant_channels: bool = False) -> None:
        """
        Ensure that the input tensor has the expected number of channels.

        The input tensor is expected to be in the form (batch[optional], channels, timestep, height, width).
        For example, a 4D tensor (channels, timestep, height, width) will have the channel dimension
        at index 0 (i.e. x.size(x.dim()-4)), and a 5D tensor (batch, channels, timestep, height, width)
        will have the channel dimension at index 1.
        """
        if x.dim() < 4:
            raise ValueError("Input tensor must have at least 4 dimensions (e.g. (C, T, H, W)).")
        ch_dim = x.size(x.dim() - 4)
        if ch_dim != self.n_channels - (self.n_constant_channels if ignore_constant_channels else 0):
            raise ValueError(
                f"Input tensor has {ch_dim} channels at index -4, expected {self.n_channels} channels."
            )

    def _reshape_buffer(self, x: torch.Tensor, buffer: torch.Tensor, ignore_constant_channels: bool = False) -> torch.Tensor:
        """
        Reshape a 1D buffer to be broadcastable with x.

        The buffer (which has shape (C,)) is reshaped to have 1 in every dimension except
        in the channel dimension (determined as x.dim() - 4).
        """
        dims = x.dim()
        shape = [1] * dims
        channel_idx = dims - 4
        shape[channel_idx] = self.n_channels - (self.n_constant_channels if ignore_constant_channels else 0)
        return buffer.view(shape)

    def forward(self, x: torch.Tensor, ignore_constant_channels: bool = False) -> torch.Tensor:
        """
        Normalize a torch tensor using the stored statistics.

        Args:
            x (torch.Tensor): Input tensor with shape either (C, H, W) or (B, C, H, W),
                              where the channel dimension is the 4th from last (i.e.,
                              x.dim()-3).

        Returns:
            torch.Tensor: Normalized tensor.
        """
        self._validate_tensor_shape(x, ignore_constant_channels)
        if not ignore_constant_channels:
            scale = self._reshape_buffer(x, self.scale, ignore_constant_channels)
            offset = self._reshape_buffer(x, self.offset, ignore_constant_channels)
        else:
            scale = self._reshape_buffer(x, self.constant_scale, ignore_constant_channels)
            offset = self._reshape_buffer(x, self.constant_offset, ignore_constant_channels)
        return x * scale + offset

    def inverse_norm(self, x: torch.Tensor, ignore_constant_channels: bool = False) -> torch.Tensor:
        """
        Denormalize a torch tensor using the stored statistics.

        Args:
            x (torch.Tensor): Normalized tensor with shape (C, H, W) or (B, C, H, W).

        Returns:
            torch.Tensor: Denormalized tensor.
        """
        self._validate_tensor_shape(x, ignore_constant_channels)
        if not ignore_constant_channels:
            scale = self._reshape_buffer(x, self.scale, ignore_constant_channels)
            offset = self._reshape_buffer(x, self.offset, ignore_constant_channels)
        else:
            scale = self._reshape_buffer(x, self.constant_scale, ignore_constant_channels)
            offset = self._reshape_buffer(x, self.constant_offset, ignore_constant_channels)
        return (x - offset) / scale


def load_torch_normalizer(
    stats_path: Union[str, Path],
    normalization_type: str = "standard"
) -> TorchNormalizationApplier:
    """
    Helper to create a TorchNormalizationApplier from a statistics file.

    Args:
        stats_path (Union[str, Path]): Path to the JSON file with statistics.
        normalization_type (str): Either "minmax" or "standard".
    
    Returns:
        TorchNormalizationApplier: The normalization module ready for use.
    """
    return TorchNormalizationApplier(stats_path, normalization_type)
