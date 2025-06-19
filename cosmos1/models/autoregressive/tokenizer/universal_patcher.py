from typing import Iterable, List
from functools import lru_cache
import numpy as np
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
from cosmos1.models.autoregressive.tokenizer.patching import Patcher3D, Patcher
from cosmos1.models.autoregressive.tokenizer.cnn_blocks import (
    DownBlock, Downsample, MiddleBlock, UpBlock, Upsample, get_activation
)


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class CrossAttnPatcher(nn.Module):
    def __init__(
        self,
        variables,
        video_size,
        patch_size=2,
        patch_method="haar",
        embed_dim=1024,
        num_heads=16,
    ):
        super().__init__()

        self.video_size = video_size
        self.patch_size = patch_size
        self.t_patches = 1 + (video_size[0]-1) // patch_size
        self.h_patches = video_size[1] // patch_size
        self.w_patches = video_size[2] // patch_size
        self.num_patches = self.t_patches * self.h_patches * self.w_patches
        self.variables = variables

        # variable tokenization: separate embedding layer for each input variable
        self.token_embeds = nn.ModuleList(
            [Patcher3D(patch_size, patch_method) for i in range(len(variables))]
        )
        self.linear_emb = nn.ModuleList(
            [nn.Linear(patch_size ** 3, embed_dim) for i in range(len(variables))]
        )

        # variable embedding to denote which variable each token belongs to
        # helps in aggregating variables
        self.channel_embed, self.channel_map = self.create_var_embedding(embed_dim)

        # variable aggregation: a learnable query and a single-layer cross attention
        self.channel_query = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)
        self.channel_agg = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        self.norm = nn.LayerNorm(embed_dim)

        self.initialize_weights()

    def initialize_weights(self):
        channel_embed = get_1d_sincos_pos_embed_from_grid(self.channel_embed.shape[-1], np.arange(len(self.variables)))
        self.channel_embed.data.copy_(torch.from_numpy(channel_embed).float().unsqueeze(0))

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def create_var_embedding(self, dim):
        var_embed = nn.Parameter(torch.zeros(1, len(self.variables), dim), requires_grad=True)
        var_map = {}
        idx = 0
        for var in self.variables:
            var_map[var] = idx
            idx += 1
        return var_embed, var_map

    @lru_cache(maxsize=None)
    def get_var_ids(self, vars, device):
        ids = np.array([self.channel_map[var] for var in vars])
        return torch.from_numpy(ids).to(device)

    def aggregate_variables(self, x: torch.Tensor):
        """
        x: B, V, L, D
        """
        b, _, l, _ = x.shape
        x = torch.einsum("bvld->blvd", x)
        x = x.flatten(0, 1)  # BxL, V, D

        var_query = self.channel_query.repeat_interleave(x.shape[0], dim=0)
        x, _ = self.channel_agg(var_query, x, x)  # BxL, D
        x = x.squeeze()

        x = x.unflatten(dim=0, sizes=(b, l))  # B, L, D
        return x

    def forward(self, x: torch.Tensor, variables):
        if isinstance(variables, list):
            variables = tuple(variables)
        
        _, _, T, H, W = x.shape
        # tokenize each variable separately
        embeds = []
        var_ids = self.get_var_ids(variables, x.device)

        for i in range(len(var_ids)):
            id = var_ids[i]
            variable_patches = self.token_embeds[id](x[:, i:i+1]) # B, p*3, t, h, w
            variable_patches = variable_patches.permute(0, 2, 3, 4, 1)  # B, t, h, w, p*3
            variable_patches = variable_patches.flatten(1, 3)  # B, t*h*w, p*3
            embed_variable = self.linear_emb[id](variable_patches)  # B, t*h*w, D
            embeds.append(embed_variable)
        x = torch.stack(embeds, dim=1)  # B, V, L, D

        # add variable embedding
        var_embed = self.channel_embed[:, var_ids, :]
        x = x + var_embed.unsqueeze(2)

        # variable aggregation
        x = self.aggregate_variables(x)  # B, L, D
        x = self.norm(x)
        
        # reshape back to B, t, h, w, D
        # print(T,H,W,x.shape)
        x = x.unflatten(dim=1, sizes=(1 + (T-1)//self.patch_size, H//self.patch_size, W//self.patch_size)) # B, t, h, w, D
        # except:
        #     breakpoint()
        x = x.permute(0, 4, 1, 2, 3)  # B, D, t, h, w

        return x, var_ids


class PaddedPatcher3D(Patcher3D):
    """A 3D discrete wavelet transform for video data, expects 5D tensor, i.e. a batch of videos."""
    def __init__(
        self,
        variables,
        patch_size=1,
        patch_method="haar",
        max_img_size=[512, 512],
        learnable_padding=False,
    ):
        super().__init__(patch_method=patch_method, patch_size=patch_size)
        self.variables = variables
        self.learnable_padding = learnable_padding
        self.max_img_size = max_img_size
        
        self.channel_map = {}
        idx = 0
        for var in self.variables:
            self.channel_map[var] = idx
            idx += 1
        
        if learnable_padding:
            self.padding = nn.Parameter(
                torch.zeros(len(variables), max_img_size[0], max_img_size[1]),
                requires_grad=True,
            )
    
    @lru_cache(maxsize=None)
    def get_var_ids(self, vars, device, get_non_ids=False):
        ids = np.array([self.channel_map[var] for var in vars])
        if not get_non_ids:
            return torch.from_numpy(ids).to(device)
        non_ids = np.array([self.channel_map[var] for var in self.variables if var not in vars])
        return torch.from_numpy(ids).to(device), torch.from_numpy(non_ids).to(device)
    
    def forward(self, x: torch.Tensor, variables):
        b, _, t, h, w = x.shape
        if isinstance(variables, list):
            variables = tuple(variables)
        
        var_ids, non_var_ids = self.get_var_ids(variables, x.device, get_non_ids=True)
        x_full_vars = torch.zeros(b, len(self.variables), t, h, w, device=x.device, dtype=x.dtype)
        x_full_vars[:, var_ids, :, :, :] = x[:, :, :, :, :]

        if self.learnable_padding:
            padding = self.padding.unsqueeze(0).expand(b, -1, -1, -1)
            # interpolate to x size
            padding = nn.functional.interpolate(
                padding,
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            ) # B, V, H, W
            padding = padding.unsqueeze(2).expand(-1, -1, t, -1, -1) # B, V, T, H, W
            x_full_vars[:, non_var_ids, :, :, :] = padding[:, non_var_ids, :, :, :]
        
        return super().forward(x_full_vars)


class UniversalProjector(nn.Module):
    """A projector that transforms videos with C channels into 3 channels."""
    def __init__(
        self,
        variables,
        hidden_dimension=None,
        n_hidden_layers=0,
        out_dimension=3,
    ):
        super().__init__()
        self.variables = variables
        self.n_hidden_layers = n_hidden_layers
        
        hidden_dimension = hidden_dimension if n_hidden_layers > 0 else out_dimension
        self.first_projector = nn.Conv2d(
            in_channels=len(variables),
            out_channels=hidden_dimension,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        if n_hidden_layers > 0:
            hidden_projectors = nn.ModuleList([nn.SiLU()])
            for i in range(n_hidden_layers-1):
                hidden_projectors.append(nn.Conv2d(
                    in_channels=hidden_dimension,
                    out_channels=hidden_dimension,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                ))
                hidden_projectors.append(nn.SiLU())
            hidden_projectors.append(
                nn.Conv2d(
                    in_channels=hidden_dimension,
                    out_channels=out_dimension,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            self.hidden_projectors = nn.Sequential(*hidden_projectors)
        else:
            self.hidden_projectors = nn.Identity()
        
        # get variable mapping
        self.channel_map = {}
        idx = 0
        for var in self.variables:
            self.channel_map[var] = idx
            idx += 1
    
    @lru_cache(maxsize=None)
    def get_var_ids(self, vars, device):
        ids = np.array([self.channel_map[var] for var in vars])
        return torch.from_numpy(ids).to(device)
    
    def forward(self, x: torch.Tensor, variables):
        # x : B, C, T, H, W
        # output: B, 3, T, H, W
        b, _, t, _, _ = x.shape
        x = x.permute(0, 2, 1, 3, 4)  # B, T, C, H, W
        x = x.flatten(0, 1)  # BT, C, H, W
        
        if isinstance(variables, list):
            variables = tuple(variables)
        
        var_ids = self.get_var_ids(variables, x.device)
        
        # Dynamically construct the first projector for the input variables
        var_projector = self.first_projector.weight[:, var_ids, :, :]
        x = nn.functional.conv2d(x, var_projector, bias=self.first_projector.bias, stride=1, padding=0)
        x = self.hidden_projectors(x)
        
        x = x.unflatten(0, (b, t))
        x = x.permute(0, 2, 1, 3, 4)
        return x, var_ids


class UniversalUnetProjector(nn.Module):
    def __init__(
        self,
        variables: List[str],
        out_channels: int = 3,
        hidden_channels=64,
        activation="gelu",
        norm: bool = True,
        dropout: float = 0.0,
        ch_mults: Iterable[int] = (1, 2, 2, 4),
        is_attn: Iterable[bool] = (False, False, False, False),
        mid_attn: bool = False,
        n_blocks: int = 2,
    ) -> None:
        super().__init__()
        self.in_channels = len(variables)
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels

        self.activation = get_activation(activation)
        self.image_proj = UniversalProjector(
            variables=variables,
            out_dimension=hidden_channels,
            hidden_dimension=None,
            n_hidden_layers=0,
        )  # project from any channels to hidden_channels
        
        for i in range(1, len(ch_mults)):
            ch_mults[i] = int(ch_mults[i] / ch_mults[i-1])

        # #### First half of U-Net - decreasing resolution
        down = []
        # Number of channels
        out_channels = in_channels = self.hidden_channels
        # For each resolution
        n_resolutions = len(ch_mults)
        for i in range(n_resolutions):
            # Number of output channels at this resolution
            out_channels = in_channels * ch_mults[i]
            # Add `n_blocks`
            for _ in range(n_blocks):
                down.append(
                    DownBlock(
                        in_channels,
                        out_channels,
                        has_attn=is_attn[i],
                        activation=activation,
                        norm=norm,
                        dropout=dropout,
                    )
                )
                in_channels = out_channels
            # Down sample at all resolutions except the last
            if i < n_resolutions - 1:
                down.append(Downsample(in_channels))

        # Combine the set of modules
        self.down = nn.ModuleList(down)

        # Middle block
        self.middle = MiddleBlock(
            out_channels,
            has_attn=mid_attn,
            activation=activation,
            norm=norm,
            dropout=dropout,
        )

        # #### Second half of U-Net - increasing resolution
        up = []
        # Number of channels
        in_channels = out_channels
        # For each resolution
        for i in reversed(range(n_resolutions)):
            # `n_blocks` at the same resolution
            out_channels = in_channels
            for _ in range(n_blocks):
                up.append(
                    UpBlock(
                        in_channels,
                        out_channels,
                        has_attn=is_attn[i],
                        activation=activation,
                        norm=norm,
                        dropout=dropout,
                    )
                )
            # Final block to reduce the number of channels
            out_channels = in_channels // ch_mults[i]
            up.append(
                UpBlock(
                    in_channels,
                    out_channels,
                    has_attn=is_attn[i],
                    activation=activation,
                    norm=norm,
                    dropout=dropout,
                )
            )
            in_channels = out_channels
            # Up sample at all resolutions except last
            if i > 0:
                up.append(Upsample(in_channels))

        # Combine the set of modules
        self.up = nn.ModuleList(up)

        if norm:
            self.norm = nn.GroupNorm(8, in_channels)
        else:
            self.norm = nn.Identity()
        self.final = nn.Conv2d(in_channels, self.out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, variables) -> torch.Tensor:
        b, _, t, _, _ = x.shape

        x, var_ids = self.image_proj(x, variables)
        
        x = x.permute(0, 2, 1, 3, 4)  # B, T, D, H, W
        x = x.flatten(0, 1)  # BT, D, H, W
        
        h = [x]
        for m in self.down:
            x = m(x)
            h.append(x)
        x = self.middle(x)
        for m in self.up:
            if isinstance(m, Upsample):
                x = m(x)
            else:
                # Get the skip connection from first half of U-Net and concatenate
                s = h.pop()
                x = torch.cat((x, s), dim=1)
                x = m(x)

        x = self.final(self.activation(self.norm(x)))
        
        x = x.unflatten(0, (b, t))
        x = x.permute(0, 2, 1, 3, 4)
        
        return x, var_ids


# patcher = CrossAttnPatcher(
#     variables=["temperature", "pressure", "humidity"],
#     video_size=(33, 512, 256),
#     patch_size=4,
#     patch_method="rearrange",
#     embed_dim=1024,
#     num_heads=4,
# ).cuda()
# x = torch.randn(1, 3, 33, 512, 256).cuda()
# variables = ["temperature", "pressure", "humidity"]
# patch_embeddings = patcher(x, variables)
# print(patch_embeddings.shape)

# patcher = PaddedPatcher3D(
#     variables=["temperature", "pressure", "humidity"],
#     patch_size=4,
#     patch_method="haar",
# ).cuda()
# x = torch.randn(1, 2, 33, 512, 256).cuda()
# variables = ["pressure", "humidity"]
# patch_embeddings = patcher(x, variables)
# print(patch_embeddings.shape)

# projector = UniversalProjector(
#     variables=["temperature", "pressure", "humidity", "velocity_x", "velocity_y"],
#     hidden_dimension=128,
#     n_hidden_layers=1,
#     out_dimension=3,
# )
# x = torch.randn(1, 4, 33, 512, 256)
# variables = ["temperature", "humidity", "velocity_x", "velocity_y"]
# output = projector(x, variables)

# unet_projector = UniversalUnetProjector(
#     variables=["temperature", "pressure", "humidity", "velocity_x", "velocity_y"],
#     out_channels=3,
#     hidden_channels=32,
#     activation="gelu",
#     norm=True,
#     dropout=0.0,
#     ch_mults=[1, 2, 4],
#     is_attn=[False, False, False],
#     mid_attn=False,
#     n_blocks=2,
# ).cuda()
# x = torch.randn(1, 4, 13, 512, 256).cuda()
# variables = ["temperature", "humidity", "velocity_x", "velocity_y"]
# x, var_ids = unet_projector(x, variables)
# print(x.shape)
# print(var_ids)
