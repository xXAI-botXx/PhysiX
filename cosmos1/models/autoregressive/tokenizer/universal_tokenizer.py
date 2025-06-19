from collections import namedtuple
from typing import List
import torch
from torch import nn

from cosmos1.models.autoregressive.tokenizer.universal_patcher import CrossAttnPatcher, UniversalProjector, UniversalUnetProjector
from cosmos1.models.autoregressive.tokenizer.modules import CausalConv3d, DecoderFactorized, EncoderFactorized
from cosmos1.models.autoregressive.tokenizer.quantizers import FSQuantizer
from cosmos1.utils import log

NetworkEval = namedtuple("NetworkEval", ["reconstructions", "quant_loss", "quant_info"])


class UniversalCausalDiscreteVideoTokenizer(nn.Module):
    def __init__(
        self,
        variables: List,
        patcher_type: str,
        z_channels: int,
        z_factor: int,
        embedding_dim: int,
        **kwargs
    ):
        super().__init__()
        self.name = kwargs.get("name", "UniversalCausalDiscreteVideoTokenizer")
        self.embedding_dim = embedding_dim
        self.patcher_type = patcher_type
        
        if patcher_type == 'cross_attn':
            self.patcher = CrossAttnPatcher(
                variables=variables,
                video_size=kwargs.get("max_video_size"),
                patch_size=kwargs.get("patch_size"),
                patch_method=kwargs.get("patch_method"),
                embed_dim=kwargs.get("patch_emb_dim"),
                num_heads=kwargs.get("patch_emb_nheads"),
            )
        elif patcher_type == 'padded':
            kwargs["padded_patcher"] = True
            kwargs["max_img_size"] = kwargs.get("max_video_size")[-2:]
            kwargs["learnable_padding"] = kwargs.get("learnable_padding")
        elif patcher_type == 'projector':
            self.projector = UniversalProjector(
                variables=variables,
                hidden_dimension=kwargs.get("hidden_dimension"),
                n_hidden_layers=kwargs.get("n_hidden_layers"),
                out_dimension=3,
            )
        elif patcher_type == 'unet_projector':
            self.projector = UniversalUnetProjector(
                variables=variables,
                out_channels=3,
                hidden_channels=kwargs.get("hidden_channels"),
                ch_mults=kwargs.get("ch_mults"),
                is_attn=kwargs.get("is_attn"),
                mid_attn=kwargs.get("mid_attn"),
                n_blocks=kwargs.get("n_blocks"),
            )
        else:
            raise ValueError(f"Unsupported patcher type: {patcher_type}")

        if patcher_type == 'cross_attn':
            modified_in_channels = kwargs.get("patch_emb_dim") // (kwargs.get("patch_size"))**3
        elif patcher_type == 'padded':
            modified_in_channels = len(variables)
        else:
            modified_in_channels = None

        kwargs["variables"] = variables
        self.encoder = EncoderFactorized(
            z_channels=z_factor * z_channels,
            in_channels=3,
            modified_in_channels=modified_in_channels,
            **kwargs
        )
        self.decoder = DecoderFactorized(
            z_channels=z_channels,
            out_channels=3,
            modified_out_channels=len(variables),
            **kwargs
        )

        self.quant_conv = CausalConv3d(z_factor * z_channels, embedding_dim, kernel_size=1, padding=0)
        self.post_quant_conv = CausalConv3d(embedding_dim, z_channels, kernel_size=1, padding=0)

        self.quantizer = FSQuantizer(**kwargs)

        num_parameters = sum(param.numel() for param in self.parameters())
        log.debug(f"model={self.name}, num_parameters={num_parameters:,}")
        log.debug(f"z_channels={z_channels}, embedding_dim={self.embedding_dim}.")

    def to(self, *args, **kwargs):
        setattr(self.quantizer, "dtype", kwargs.get("dtype", torch.bfloat16))
        return super(UniversalCausalDiscreteVideoTokenizer, self).to(*args, **kwargs)

    def encode(self, x, variables):
        # x: [B, T, H, W, C]
        if self.patcher_type == 'cross_attn':
            x, var_ids = self.patcher(x, variables) # B, D, t, h, w
            h = self.encoder(x, skip_patcher=True)
        elif self.patcher_type in ['projector', 'unet_projector']:
            x, var_ids = self.projector(x, variables)
            h = self.encoder(x)
        else:
            var_ids = self.encoder.patcher3d.get_var_ids(tuple(variables), x.device)
            h = self.encoder(x, skip_patcher=False, variables=variables)
        h = self.quant_conv(h)
        return self.quantizer(h), var_ids

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        return self.decoder(quant)

    def forward(self, input, variables):
        (quant_info, quant_codes, quant_loss), var_ids = self.encode(input, variables)
        reconstructions = self.decode(quant_codes)
        reconstructions = reconstructions[:, var_ids]
        if self.training:
            return dict(reconstructions=reconstructions, quant_loss=quant_loss, quant_info=quant_info)
        return NetworkEval(reconstructions=reconstructions, quant_loss=quant_loss, quant_info=quant_info)


# model = UniversalCausalDiscreteVideoTokenizer(
#     variables=[
#         "var_1",
#         "var_2",
#         "var_3",
#         "var_4",
#         "var_5",
#         "var_6",
#         "var_7",
#         "var_8",
#         "var_9",
#         "var_10",
#     ],
#     # patcher_type="cross_attn",
#     # max_video_size=[33, 1024, 512],
#     # patch_emb_dim=1024,
#     # patch_emb_nheads=16,
#     # patcher_type="padded",
#     # patcher_type="projector",
#     # hidden_dimension=128,
#     # n_hidden_layers=2,
#     patcher_type="unet_projector",
#     hidden_channels=128,
#     ch_mults=[1, 2, 4],
#     is_attn=[False, False, False],
#     mid_attn=False,
#     n_blocks=2,
#     z_channels=16,
#     z_factor=1,
#     patch_size=4,
#     patch_method="rearrange",
#     channels=128,
#     channels_mult=[1,2,4],
#     num_res_blocks=2,
#     attn_resolutions=[],
#     dropout=0.0,
#     resolution=1024,
#     spatial_compression=8,
#     temporal_compression=4,
#     legacy_mode=False,
#     levels=[8, 8, 8, 5, 5, 5],
#     embedding_dim=6,
# ).cuda()
# x = torch.randn(1, 5, 13, 256, 256).cuda()
# variables = ["var_1", "var_3", "var_5", "var_7", "var_9"]
# output = model(x, variables)
# print (output['reconstructions'].shape)
