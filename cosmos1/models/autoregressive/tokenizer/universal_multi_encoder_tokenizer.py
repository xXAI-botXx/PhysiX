from collections import namedtuple
from typing import List, Dict
import torch
from torch import nn
from functools import lru_cache
import numpy as np

from cosmos1.models.autoregressive.tokenizer.modules import CausalConv3d, DecoderFactorized, EncoderFactorizedDownOnly, EncoderFactorizedMidOnly
from cosmos1.models.autoregressive.tokenizer.quantizers import FSQuantizer
from cosmos1.utils import log

NetworkEval = namedtuple("NetworkEval", ["reconstructions", "quant_loss", "quant_info"])


class UniversalMultiEncoderCausalDiscreteVideoTokenizer(nn.Module):
    def __init__(
        self,
        data_metadata: Dict,
        z_channels: int,
        z_factor: int,
        embedding_dim: int,
        **kwargs
    ):
        super().__init__()
        self.name = kwargs.get("name", "UniversalMultiNetworkCausalDiscreteVideoTokenizer")
        self.embedding_dim = embedding_dim
        
        self.encoder_down = nn.ModuleDict({
            dataset: EncoderFactorizedDownOnly(
                z_channels=None,
                in_channels=3,
                modified_in_channels=len(data_metadata[dataset]["channel_names"]),
                **kwargs
            )
            for dataset in data_metadata.keys()
        })
        
        self.encoder_mid = EncoderFactorizedMidOnly(
            z_channels=z_channels,
            in_channels=None,
            modified_in_channels=None,
            **kwargs
        )
        
        variables = []
        for dataset in data_metadata.keys():
            variables.extend(data_metadata[dataset]["channel_names"])
        self.variables = variables
        
        self.channel_map = {}
        idx = 0
        for var in self.variables:
            self.channel_map[var] = idx
            idx += 1

        self.decoder = DecoderFactorized(
            z_channels=z_channels,
            out_channels=3,
            modified_out_channels=len(self.variables),
            **kwargs
        )

        self.quant_conv = CausalConv3d(z_factor * z_channels, embedding_dim, kernel_size=1, padding=0)
        self.post_quant_conv = CausalConv3d(embedding_dim, z_channels, kernel_size=1, padding=0)

        self.quantizer = FSQuantizer(**kwargs)

        num_parameters = sum(param.numel() for param in self.parameters())
        log.debug(f"model={self.name}, num_parameters={num_parameters:,}")
        log.debug(f"z_channels={z_channels}, embedding_dim={self.embedding_dim}.")
    
    @lru_cache(maxsize=None)
    def get_var_ids(self, vars, device, get_non_ids=False):
        ids = np.array([self.channel_map[var] for var in vars])
        if not get_non_ids:
            return torch.from_numpy(ids).to(device)
        non_ids = np.array([self.channel_map[var] for var in self.variables if var not in vars])
        return torch.from_numpy(ids).to(device), torch.from_numpy(non_ids).to(device)

    def to(self, *args, **kwargs):
        setattr(self.quantizer, "dtype", kwargs.get("dtype", torch.bfloat16))
        return super(UniversalMultiEncoderCausalDiscreteVideoTokenizer, self).to(*args, **kwargs)

    def encode(self, x, dataset_name):
        h = self.encoder_down[dataset_name](x)
        h = self.encoder_mid(h)
        h = self.quant_conv(h)
        return self.quantizer(h)

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        return self.decoder(quant)

    def forward(self, input, dataset_name, variables):
        quant_info, quant_codes, quant_loss = self.encode(input, dataset_name)
        reconstructions = self.decode(quant_codes)
        var_ids = self.get_var_ids(tuple(variables), reconstructions.device)
        reconstructions = reconstructions[:, var_ids]
        if self.training:
            return dict(reconstructions=reconstructions, quant_loss=quant_loss, quant_info=quant_info)
        return NetworkEval(reconstructions=reconstructions, quant_loss=quant_loss, quant_info=quant_info)