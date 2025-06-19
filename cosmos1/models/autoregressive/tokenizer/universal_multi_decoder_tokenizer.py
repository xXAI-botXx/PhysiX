from collections import namedtuple
from typing import List, Dict
import torch
from torch import nn

from cosmos1.models.autoregressive.tokenizer.modules import CausalConv3d, EncoderFactorized, DecoderFactorized
from cosmos1.models.autoregressive.tokenizer.quantizers import FSQuantizer
from cosmos1.utils import log

NetworkEval = namedtuple("NetworkEval", ["reconstructions", "quant_loss", "quant_info"])


class UniversalMultiDecoderCausalDiscreteVideoTokenizer(nn.Module):
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
        self.data_metadata = data_metadata
        
        variables = []
        for dataset in data_metadata.keys():
            variables.extend(data_metadata[dataset]["channel_names"])
        self.variables = variables
        
        self.channel_map = {}
        idx = 0
        for var in self.variables:
            self.channel_map[var] = idx
            idx += 1
        
        kwargs["padded_patcher"] = True
        kwargs["max_img_size"] = kwargs.get("max_video_size")[-2:]
        kwargs["learnable_padding"] = kwargs.get("learnable_padding")
        kwargs["variables"] = variables
        kwargs["n_datasets"] = len(data_metadata)
        self.encoder = EncoderFactorized(
            z_channels=z_factor * z_channels,
            in_channels=3,
            modified_in_channels=len(variables),
            **kwargs
        )

        self.decoder = nn.ModuleDict({
            dataset: DecoderFactorized(
                z_channels=z_channels,
                out_channels=3,
                modified_out_channels=len(self.variables),
                **kwargs
            )
            for dataset in data_metadata.keys()
        })

        self.quant_conv = CausalConv3d(z_factor * z_channels, embedding_dim, kernel_size=1, padding=0)
        self.post_quant_conv = CausalConv3d(embedding_dim, z_channels, kernel_size=1, padding=0)

        self.quantizer = FSQuantizer(**kwargs)
        self.film = kwargs.get("film", False)

        num_parameters = sum(param.numel() for param in self.parameters())
        log.debug(f"model={self.name}, num_parameters={num_parameters:,}")
        log.debug(f"z_channels={z_channels}, embedding_dim={self.embedding_dim}.")

    def to(self, *args, **kwargs):
        setattr(self.quantizer, "dtype", kwargs.get("dtype", torch.bfloat16))
        return super(UniversalMultiDecoderCausalDiscreteVideoTokenizer, self).to(*args, **kwargs)

    def encode(self, x, dataset_name, variables, checkpointing=False):
        if self.film:
            dataset_id = list(self.data_metadata.keys()).index(dataset_name)
            dataset_id = torch.tensor([dataset_id]*x.shape[0], device=x.device, dtype=torch.long)
        else:
            dataset_id = None
        var_ids = self.encoder.patcher3d.get_var_ids(tuple(variables), x.device)
        h = self.encoder(x, skip_patcher=False, variables=variables, dataset_id=dataset_id, checkpointing=checkpointing)
        h = self.quant_conv(h)
        return self.quantizer(h), var_ids

    def decode(self, quant, dataset_name, checkpointing=False):
        quant = self.post_quant_conv(quant)
        return self.decoder[dataset_name](quant, checkpointing=checkpointing)

    def forward(self, input, dataset_name, variables, checkpointing=False):
        (quant_info, quant_codes, quant_loss), var_ids = self.encode(input, dataset_name, variables, checkpointing=checkpointing)
        reconstructions = self.decode(quant_codes, dataset_name, checkpointing=checkpointing)
        reconstructions = reconstructions[:, var_ids]
        if self.training:
            return dict(reconstructions=reconstructions, quant_loss=quant_loss, quant_info=quant_info)
        return NetworkEval(reconstructions=reconstructions, quant_loss=quant_loss, quant_info=quant_info)