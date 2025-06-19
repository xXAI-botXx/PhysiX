from cosmos1.models.autoregressive.tokenizer.networks import (
    CausalDiscreteVideoTokenizer
)
from cosmos1.models.tokenizer.networks import CausalContinuousVideoTokenizer

def build_tokenizer(hparams):
    """
    Create and return a tokenizer based on the passed hyperparameters.
    """
    if hparams.mode == "continuous":
        tokenizer = CausalContinuousVideoTokenizer(
            latent_channels=hparams.latent_channels,
            formulation=hparams.formulation,
            encoder=hparams.encoder,
            decoder=hparams.decoder,
            in_channels=hparams.in_channels if hparams.scratch else 3,
            out_channels=hparams.out_channels if hparams.scratch else 3,
            channels=hparams.channels,
            channels_mult=hparams.channels_mult,
            z_channels=hparams.z_channels,
            z_factor=hparams.z_factor,
            spatial_compression=hparams.spatial_compression,
            temporal_compression=hparams.temporal_compression,
            num_res_blocks=hparams.num_res_blocks,
            patch_size=hparams.patch_size,
            patch_method=hparams.patch_method,
            num_groups=hparams.num_groups,
            resolution=hparams.resolution,
            attn_resolutions=hparams.attn_resolutions,
            dropout=hparams.dropout,
            legacy_mode=hparams.legacy_mode,
            modified_in_channels=None if hparams.scratch else hparams.in_channels,
            modified_out_channels=None if hparams.scratch else hparams.out_channels,
        )
    elif hparams.mode == "discrete":
        tokenizer = CausalDiscreteVideoTokenizer(
            z_channels=hparams.z_channels,
            z_factor=hparams.z_factor,
            embedding_dim=hparams.embedding_dim,
            in_channels=hparams.in_channels if hparams.scratch else 3,
            out_channels=hparams.out_channels if hparams.scratch else 3,
            channels=hparams.channels,
            channels_mult=hparams.channels_mult,
            levels=hparams.levels,
            spatial_compression=hparams.spatial_compression,
            temporal_compression=hparams.temporal_compression,
            num_res_blocks=hparams.num_res_blocks,
            patch_size=hparams.patch_size,
            patch_method=hparams.patch_method,
            num_groups=hparams.num_groups,
            resolution=hparams.resolution,
            attn_resolutions=hparams.attn_resolutions,
            dropout=hparams.dropout,
            legacy_mode=hparams.legacy_mode,
            modified_in_channels=None if hparams.scratch else hparams.in_channels,
            modified_out_channels=None if hparams.scratch else hparams.out_channels,
        )
    else:
        raise ValueError(f"Invalid mode: {hparams.mode}")
    return tokenizer 