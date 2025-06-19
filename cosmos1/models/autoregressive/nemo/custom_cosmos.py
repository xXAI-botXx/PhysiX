import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Callable, Optional, Dict

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from nemo.lightning import get_vocab_size, io
from nemo.collections.llm.gpt.model.llama import Llama3Config, LlamaModel
from nemo.collections.llm.gpt.model.base import get_batch_on_this_context_parallel_rank, get_packed_seq_params
from nemo.collections.llm.utils import Config
from nemo.lightning import OptimizerModule, io
from nemo.lightning.base import teardown
from nemo.utils.import_utils import safe_import
from nemo.utils import logging
from torch import Tensor, nn

from cosmos1.utils import log
from cosmos1.models.autoregressive.nemo.custom_gpt_model import CustomGPTModel

_, HAVE_TE = safe_import("transformer_engine")

def custom_cosmos_data_step(dataloader_iter) -> Dict[str, torch.Tensor]:
    from megatron.core import parallel_state

    # Based on: https://github.com/NVIDIA/Megatron-LM/blob/main/pretrain_gpt.py#L87
    # https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/nlp/models/language_modeling/megatron_gpt_model.py#L828-L842

    batch = next(dataloader_iter)

    _batch: dict
    if isinstance(batch, tuple) and len(batch) == 3:
        _batch = batch[0]
    else:
        _batch = batch

    required_device_keys = set()
    required_host_keys = set()

    required_device_keys.add("attention_mask")
    required_device_keys.add("latent_shapes")
    if 'cu_seqlens' in _batch:
        required_device_keys.add('cu_seqlens')
        required_host_keys.add('cu_seqlens_argmin')
        required_host_keys.add('max_seqlen')

    if parallel_state.is_pipeline_first_stage():
        required_device_keys.update(("tokens", "position_ids"))
    if parallel_state.is_pipeline_last_stage():
        required_device_keys.update(("labels", "loss_mask"))

    _batch_required_keys = {}
    for key, val in _batch.items():
        if key in required_device_keys:
            _batch_required_keys[key] = val.cuda(non_blocking=True)
        elif key in required_host_keys:
            _batch_required_keys[key] = val.cpu()
        else:
            _batch_required_keys[key] = None

    # slice batch along sequence dimension for context parallelism
    output = get_batch_on_this_context_parallel_rank(_batch_required_keys)

    return output

def custom_cosmos_forward_step(model, batch) -> torch.Tensor:
    forward_args = {
        "input_ids": batch["tokens"],
        "position_ids": batch["position_ids"],
        "labels": batch["labels"],
        "latent_shapes": batch["latent_shapes"],
    }

    if 'attention_mask' not in batch:
        assert (
            HAVE_TE
        ), "The dataloader did not provide an attention mask, however Transformer Engine was not detected. \
            This requires Transformer Engine's implementation of fused or flash attention."
    else:
        forward_args["attention_mask"] = batch['attention_mask']

    if 'cu_seqlens' in batch:
        forward_args['packed_seq_params'] = get_packed_seq_params(batch)

    return model(**forward_args)


class CustomRotaryEmbedding3D(RotaryEmbedding):
    """Rotary Embedding3D for Cosmos Language model.
    Args:
        kv_channels (int): Projection weights dimension in multi-head attention. Obtained
            from transformer config
        rotary_base (int, optional): Base period for rotary position embeddings. Defaults to
            10000.
        use_cpu_initialization (bool, optional): If False, initialize the inv_freq directly
            on the GPU. Defaults to False
        latent_shape: The shape of the latents produced by the video after being tokenized
    """

    def __init__(
        self,
        seq_len: int,
        kv_channels: int,
        training_type: str = None,
        rotary_base: int = 10000,
        use_cpu_initialization: bool = False,
        latent_shape=[5, 40, 64],
        apply_yarn=False,
        original_latent_shape=None,
        beta_fast=32,
        beta_slow=1,
        scale=None,
        max_position_embeddings=None,
        original_max_position_embeddings=None,
        extrapolation_factor=1,
        attn_factor=1,
    ) -> None:
        super().__init__(
            kv_channels=kv_channels,
            rotary_base=rotary_base,
            rotary_percent=1.0,
            use_cpu_initialization=use_cpu_initialization,
        )
        self.latent_shape = latent_shape
        self.device = "cpu" if use_cpu_initialization else torch.cuda.current_device()
        self.dim = kv_channels
        self.rope_theta = rotary_base
        self.apply_yarn = apply_yarn
        self.original_latent_shape = original_latent_shape
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.scale = scale
        self.max_position_embeddings = max_position_embeddings
        self.original_max_position_embeddings = original_max_position_embeddings
        self.attn_factor = attn_factor
        dim_h = self.dim // 6 * 2
        dim_t = self.dim - 2 * dim_h
        self.dim_spatial_range = torch.arange(0, dim_h, 2)[: (dim_h // 2)].float().to(self.device) / dim_h
        spatial_inv_freq = 1.0 / (self.rope_theta**self.dim_spatial_range)
        self.dim_temporal_range = torch.arange(0, dim_t, 2)[: (dim_t // 2)].float().to(self.device) / dim_t
        temporal_inv_freq = 1.0 / (self.rope_theta**self.dim_temporal_range)
        if self.apply_yarn:
            assert self.original_latent_shape is not None, "Original latent shape required."
            assert self.beta_slow is not None, "Beta slow value required."
            assert self.beta_fast is not None, "Beta fast value required."
            scale_factors_spatial = self.get_scale_factors(spatial_inv_freq, self.original_latent_shape[1])
            spatial_inv_freq = spatial_inv_freq * scale_factors_spatial
            scale_factors_temporal = self.get_scale_factors(temporal_inv_freq, self.original_latent_shape[0])
            temporal_inv_freq = temporal_inv_freq * scale_factors_temporal
            self.mscale = float(self.get_mscale(self.scale) * self.attn_factor)
        self.spatial_inv_freq = spatial_inv_freq
        self.temporal_inv_freq = temporal_inv_freq
        max_seq_len_cached = max(self.latent_shape)
        if self.apply_yarn and seq_len > max_seq_len_cached:
            max_seq_len_cached = seq_len
        self.max_seq_len_cached = max_seq_len_cached
        self.freqs = self.get_freqs_non_repeated(self.max_seq_len_cached)

    def get_mscale(self, scale: float = 1.0) -> float:
        """Get the magnitude scaling factor for YaRN."""
        if scale <= 1:
            return 1.0
        return 0.1 * math.log(scale) + 1.0

    def get_scale_factors(self, inv_freq: torch.Tensor, original_seq_len: int) -> torch.Tensor:
        """Get the scale factors for YaRN."""
        # Calculate the high and low frequency cutoffs for YaRN. Note: `beta_fast` and `beta_slow` are called
        # `high_freq_factor` and `low_freq_factor` in the Llama 3.1 RoPE scaling code.
        high_freq_cutoff = 2 * math.pi * self.beta_fast / original_seq_len
        low_freq_cutoff = 2 * math.pi * self.beta_slow / original_seq_len
        # Obtain a smooth mask that has a value of 0 for low frequencies and 1 for high frequencies, with linear
        # interpolation in between.
        smooth_mask = torch.clamp((inv_freq - low_freq_cutoff) / (high_freq_cutoff - low_freq_cutoff), min=0, max=1)
        # For low frequencies, we scale the frequency by 1/self.scale. For high frequencies, we keep the frequency.
        scale_factors = (1 - smooth_mask) / self.scale + smooth_mask
        return scale_factors

    def get_freqs_non_repeated(self, max_seq_len_cached: int, latent_shapes: Tensor = None, offset: int = 0) -> Tensor:
        dtype = self.spatial_inv_freq.dtype
        device = self.spatial_inv_freq.device

        self.seq = (torch.arange(max_seq_len_cached, device=device, dtype=dtype) + offset).cuda()

        assert hasattr(
            self, "latent_shape"
        ), "Latent shape is not set. Please run set_latent_shape() method on rope embedding. "
        T, H, W = self.latent_shape
        if latent_shapes is not None:
            T, H, W = latent_shapes[0, 0].item(), latent_shapes[0, 1].item(), latent_shapes[0, 2].item()
        half_emb_t = torch.outer(self.seq[:T], self.temporal_inv_freq.cuda())
        half_emb_h = torch.outer(self.seq[:H], self.spatial_inv_freq.cuda())
        half_emb_w = torch.outer(self.seq[:W], self.spatial_inv_freq.cuda())
        emb = torch.cat(
            [
                repeat(half_emb_t, "t d -> t h w d", h=H, w=W),
                repeat(half_emb_h, "h d -> t h w d", t=T, w=W),
                repeat(half_emb_w, "w d -> t h w d", t=T, h=H),
            ]
            * 2,
            dim=-1,
        )
        emb = rearrange(emb, "t h w d -> (t h w) 1 1 d").float()
        return emb

    @lru_cache(maxsize=32)
    def forward(self, seq_len: int, latent_shapes: Tensor = None, offset: int = 0, packed_seq: bool = False) -> Tensor:
        if self.spatial_inv_freq.device.type == "cpu":
            # move `inv_freq` to GPU once at the first micro-batch forward pass
            self.spatial_inv_freq = self.spatial_inv_freq.to(device=torch.cuda.current_device())

        max_seq_len_cached = self.max_seq_len_cached
        if self.apply_yarn and seq_len > max_seq_len_cached:
            max_seq_len_cached = seq_len
        self.max_seq_len_cached = max_seq_len_cached
        emb = self.get_freqs_non_repeated(self.max_seq_len_cached, latent_shapes)
        return emb


class LuminaCustomRotaryEmbedding3D(RotaryEmbedding):
    """
    API-compatible replacement that *recomputes* RoPE frequencies per latent
    shape so every axis spans its full angular range, following ResFormer /
    FlexiViT multi-resolution training.

    All YaRN-related arguments are kept but only used when `apply_yarn=True`.
    """
    def __init__(
        self,
        seq_len: int,
        kv_channels: int,
        training_type: str = None,
        rotary_base: int = 10000,
        use_cpu_initialization: bool = False,
        latent_shape=[5, 40, 64],
        apply_yarn=False,
        original_latent_shape=None,
        beta_fast=32,
        beta_slow=1,
        scale=None,
        max_position_embeddings=None,
        original_max_position_embeddings=None,
        extrapolation_factor=1,
        attn_factor=1,
    ) -> None:
        super().__init__(
            kv_channels=kv_channels,
            rotary_base=rotary_base,
            rotary_percent=1.0,
            use_cpu_initialization=use_cpu_initialization,
        )
        self.latent_shape = latent_shape
        self.device = "cpu" if use_cpu_initialization else torch.cuda.current_device()
        self.dim = kv_channels
        self.rope_theta = rotary_base
        self.apply_yarn = apply_yarn
        self.original_latent_shape = original_latent_shape
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.scale = scale
        self.max_position_embeddings = max_position_embeddings
        self.original_max_position_embeddings = original_max_position_embeddings
        self.attn_factor = attn_factor
        self.dim_h = self.dim_w = self.dim // 6 * 2
        self.dim_t = self.dim - 2 * self.dim_h        
        self._ref_T, self._ref_H, self._ref_W = self.latent_shape

    def precompute_freqs_cis(
        dim: int,
        end: int,
        theta: float = 10000.0,
        scale_factor: float = 1.0,
        scale_watershed: float = 1.0,
        timestep: float = 1.0,
    ):
        if timestep < scale_watershed:
            linear_factor = scale_factor
            ntk_factor = 1.0
        else:
            linear_factor = 1.0
            ntk_factor = scale_factor

        theta = theta * ntk_factor
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 6)[: (dim // 6)] / dim)) / linear_factor

        timestep = torch.arange(end, dtype=torch.float32)
        freqs = torch.outer(timestep, freqs).float()
        freqs_cis = torch.exp(1j * freqs)

        freqs_cis_t = freqs_cis.view(end, 1, 1, dim // 6).repeat(1, end, end, 1)
        freqs_cis_h = freqs_cis.view(1, end, 1, dim // 6).repeat(end, 1, end, 1)
        freqs_cis_w = freqs_cis.view(1, 1, end, dim // 6).repeat(end, end, 1, 1)
        freqs_cis = torch.cat([freqs_cis_t, freqs_cis_h, freqs_cis_w], dim=-1).view(end, end, end, -1)
        return freqs_cis

    def _build_inv_freq(self, part_dim: int, axis_len: int, ref_len: int, device):
        """
        part_dim : even dimension size allocated to this axis (e.g. self.dim_h)
        axis_len : actual tokens on this axis for the current clip
        ref_len  : reference (max) length recorded at construction
        """
        ntk_factor = axis_len / ref_len
        theta = self.rope_theta * ntk_factor
        freqs = 1.0 / (theta ** (torch.arange(0, part_dim, 2)[: (part_dim // 2)] / part_dim))
        freqs = freqs.to(device)
        return freqs
        # idx = torch.arange(0, part_dim, 2, device=device).float()[: part_dim // 2]
        # base_inv_freq = 1.0 / (self.rope_theta ** (idx / part_dim))
        # # scale frequencies so that full axis length covers same angle as ref_len
        # return base_inv_freq * (ref_len / axis_len)

    # ------------------------------------------------------------
    # core frequency grid ------------------------------------------------------
    @lru_cache(maxsize=32)
    def get_freqs_non_repeated(self, max_seq_len_cached: int,
                               latent_shapes: torch.Tensor = None,
                               offset: int = 0) -> torch.Tensor:
        """
        Re-implements parent method but rebuilds inv_freq per (T,H,W).
        Output shape: (T*H*W, 1, 1, dim)
        """
        # ----- resolve (T,H,W) ---------
        if latent_shapes is not None:
            T, H, W = (int(x) for x in latent_shapes[0])
        else:
            T, H, W = self.latent_shape
        device = torch.cuda.current_device()

        # ----- per‑axis inv_freq, axis‑normalised -----

        inv_t = self._build_inv_freq(self.dim_t, T, self._ref_T, device)
        inv_h = self._build_inv_freq(self.dim_h, H, self._ref_H, device)
        inv_w = self._build_inv_freq(self.dim_w, W, self._ref_W, device)

        seq_t = torch.arange(T, device=device, dtype=inv_t.dtype) + offset
        seq_h = torch.arange(H, device=device, dtype=inv_h.dtype) + offset
        seq_w = torch.arange(W, device=device, dtype=inv_w.dtype) + offset

        half_t = torch.outer(seq_t, inv_t)
        half_h = torch.outer(seq_h, inv_h)
        half_w = torch.outer(seq_w, inv_w)

        emb = torch.cat(
            [
                repeat(half_t, "t d -> t h w d", h=H, w=W),
                repeat(half_h, "h d -> t h w d", t=T, w=W),
                repeat(half_w, "w d -> t h w d", t=T, h=H),
            ]
            * 2,
            dim=-1,
        )
        return rearrange(emb, "t h w d -> (t h w) 1 1 d").float()

    # ------------------------------------------------------------
    # public forward -----------------------------------------------------------
    @lru_cache(maxsize=32)
    def forward(self, seq_len: int,
                latent_shapes: torch.Tensor = None,
                offset: int = 0,
                packed_seq: bool = False) -> torch.Tensor:
        # we ignore seq_len because RoPE depends only on latent shape
        return self.get_freqs_non_repeated(
            max_seq_len_cached=0,  # dummy; not used after rewrite
            latent_shapes=latent_shapes,
            offset=offset,
        )


if TYPE_CHECKING:
    from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec


@dataclass
class CustomCosmosConfig(Llama3Config):
    qk_layernorm: bool = True
    rope_dim: str = "3D"
    vocab_size: int = 64000
    activation_func = F.silu
    
    forward_step_fn: Callable = custom_cosmos_forward_step
    data_step_fn: Callable = custom_cosmos_data_step
    
    def configure_model(self, tokenizer, pre_process=None, post_process=None) -> "CustomGPTModel":
        if self.enable_cuda_graph:
            assert HAVE_TE, "Transformer Engine is required for cudagraphs."
            assert getattr(self, 'use_te_rng_tracker', False), (
                "Transformer engine's RNG tracker is required for cudagraphs, it can be "
                "enabled with use_te_rng_tracker=True'."
            )

        vp_size = self.virtual_pipeline_model_parallel_size
        is_pipeline_asymmetric = getattr(self, 'account_for_embedding_in_pipeline_split', False) or getattr(
            self, 'account_for_loss_in_pipeline_split', False
        )
        if vp_size and not is_pipeline_asymmetric:
            p_size = self.pipeline_model_parallel_size
            assert (
                self.num_layers // p_size
            ) % vp_size == 0, "Make sure the number of model chunks is the same across all pipeline stages."

        from megatron.core import parallel_state

        transformer_layer_spec = self.transformer_layer_spec
        if not isinstance(transformer_layer_spec, ModuleSpec):
            transformer_layer_spec = transformer_layer_spec(self)

        if hasattr(self, 'vocab_size'):
            vocab_size = self.vocab_size
            if tokenizer is not None:
                logging.info(
                    f"Use preset vocab_size: {vocab_size}, original vocab_size: {tokenizer.vocab_size}, dummy tokens:"
                    f" {vocab_size - tokenizer.vocab_size}."
                )
        else:
            vocab_size = get_vocab_size(self, tokenizer.vocab_size, self.make_vocab_size_divisible_by)

        model = CustomGPTModel(
            self,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=vocab_size,
            max_sequence_length=self.seq_length,
            fp16_lm_cross_entropy=self.fp16_lm_cross_entropy,
            parallel_output=self.parallel_output,
            share_embeddings_and_output_weights=self.share_embeddings_and_output_weights,
            position_embedding_type=self.position_embedding_type,
            rotary_percent=self.rotary_percent,
            rotary_base=self.rotary_base,
            seq_len_interpolation_factor=self.seq_len_interpolation_factor,
            pre_process=pre_process or parallel_state.is_pipeline_first_stage(),
            post_process=post_process or parallel_state.is_pipeline_last_stage(),
            scatter_embedding_sequence_parallel=self.scatter_embedding_sequence_parallel,
        )
        
        if self.rope_dim == "3D":
            model.rotary_pos_emb = CustomRotaryEmbedding3D(
            # model.rotary_pos_emb = LuminaCustomRotaryEmbedding3D(
                seq_len=self.seq_length,
                training_type=None,
                kv_channels=self.kv_channels,
                max_position_embeddings=self.seq_length,
                original_max_position_embeddings=self.original_seq_len if hasattr(self, "original_seq_len") else None,
                rotary_base=self.rotary_base,
                apply_yarn=True if hasattr(self, "apply_yarn") else False,
                scale=self.yarn_scale if hasattr(self, "yarn_scale") else None,
                extrapolation_factor=1,
                attn_factor=1,
                beta_fast=self.yarn_beta_fast if hasattr(self, "yarn_beta_fast") else 32,
                beta_slow=self.yarn_beta_slow if hasattr(self, "yarn_beta_slow") else 1,
                latent_shape=self.latent_shape if hasattr(self, "latent_shape") else [5, 16, 32],
                original_latent_shape=self.original_latent_shape if hasattr(self, "original_latent_shape") else None,
            )

        # If using full TE layer, need to set TP, CP group since the module call
        # is not routed through megatron core, which normally handles passing the
        # TP, CP group to the TE modules.
        # Deep iterate but skip self to avoid infinite recursion.
        if HAVE_TE and self.use_transformer_engine_full_layer_spec:
            # Copied from:
            # https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/transformer.py
            if parallel_state.get_tensor_model_parallel_world_size() > 1:
                for index, child in enumerate(model.modules()):
                    if index == 0:
                        continue
                    if hasattr(child, "set_tensor_parallel_group"):
                        tp_group = parallel_state.get_tensor_model_parallel_group()
                        child.set_tensor_parallel_group(tp_group)

            if parallel_state.get_context_parallel_world_size() > 1:
                cp_stream = torch.cuda.Stream()
                for module in self.get_model_module_list():
                    for index, child in enumerate(module.modules()):
                        if index == 0:
                            continue
                        if hasattr(child, "set_context_parallel_group"):
                            child.set_context_parallel_group(
                                parallel_state.get_context_parallel_group(),
                                parallel_state.get_context_parallel_global_ranks(),
                                cp_stream,
                            )

        return model

@dataclass
class CustomCosmosConfig600M(CustomCosmosConfig):
    rotary_base: int = 500_000
    seq_length: int = 15360
    num_layers: int = 8
    hidden_size: int = 2048
    ffn_hidden_size: int = 8192
    num_attention_heads: int = 16
    num_query_groups: int = 8
    layernorm_epsilon: float = 1e-5
    use_cpu_initialization: bool = True
    make_vocab_size_divisible_by: int = 128
    kv_channels: int = 128

@dataclass
class CustomCosmosConfig2B(CustomCosmosConfig):
    rotary_base: int = 500_000
    seq_length: int = 15360
    num_layers: int = 12
    hidden_size: int = 3072
    ffn_hidden_size: int = 12288
    num_attention_heads: int = 24
    num_query_groups: int = 8
    layernorm_epsilon: float = 1e-5
    use_cpu_initialization: bool = True
    make_vocab_size_divisible_by: int = 128
    kv_channels: int = 128

@dataclass
class CustomCosmosConfig4B(CustomCosmosConfig):
    rotary_base: int = 500_000
    seq_length: int = 15360
    num_layers: int = 16
    hidden_size: int = 4096
    ffn_hidden_size: int = 14336
    num_attention_heads: int = 32
    num_query_groups: int = 8
    layernorm_epsilon: float = 1e-5
    use_cpu_initialization: bool = True
    make_vocab_size_divisible_by: int = 128
    kv_channels: int = 128


@dataclass
class CustomCosmosConfig12B(CustomCosmosConfig):
    rotary_base: int = 500_000
    seq_length: int = 15360
    num_layers: int = 40
    hidden_size: int = 5120
    ffn_hidden_size: int = 14336
    num_attention_heads: int = 32
    num_query_groups: int = 8
    layernorm_epsilon: float = 1e-5
    use_cpu_initialization: bool = True
    make_vocab_size_divisible_by: int = 128
    kv_channels: int = 128
    original_latent_shape = [3, 40, 64]
    apply_yarn: bool = True
    yarn_beta_fast: int = 4
    yarn_beta_slow: int = 1
    yarn_scale: int = 2
    original_seq_len = 8192


class CustomCosmosModel(LlamaModel):
    def __init__(
        self,
        config: Annotated[Optional[CustomCosmosConfig], Config[CustomCosmosConfig]] = None,
        optim: Optional[OptimizerModule] = None,
        tokenizer: Optional["TokenizerSpec"] = None,
        model_transform: Optional[Callable[[nn.Module], nn.Module]] = None,
    ):
        super().__init__(config or CustomCosmosConfig4B(), optim=optim, tokenizer=tokenizer, model_transform=model_transform)
        self.config = config
    
    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        latent_shapes: Optional[torch.Tensor] = None,
        decoder_input: Optional[torch.Tensor] = None,
        inference_params=None,
        packed_seq_params=None,
    ) -> torch.Tensor:
        extra_kwargs = {'packed_seq_params': packed_seq_params} if packed_seq_params is not None else {}
        output_tensor = self.module(
            input_ids,
            position_ids,
            attention_mask,
            decoder_input=decoder_input,
            labels=labels,
            latent_shapes=latent_shapes,
            inference_params=inference_params,
            **extra_kwargs,
        )

        return output_tensor
    
    def training_step(self, batch, batch_idx=None) -> torch.Tensor:
        # In mcore the loss-function is part of the forward-pass (when labels are provided)

        return self.forward_step(batch)
    
    def on_validation_epoch_start(self):
        self.agg_loss = []
    
    def validation_step(self, batch, batch_idx=None) -> torch.Tensor:
        # In mcore the loss-function is part of the forward-pass (when labels are provided)
        loss = self.forward_step(batch)
        self.agg_loss.append(loss.mean().item())
        
        return loss

    def on_validation_epoch_end(self):
        # aggregate self.agg_loss across all processes
        local_mean_loss = torch.tensor(sum(self.agg_loss) / len(self.agg_loss)).to(self.device)
        
        torch.distributed.barrier()
        
        gathered_means = self.all_gather(local_mean_loss)
        # Calculate the global mean across all ranks
        global_mean_loss = gathered_means.mean().item()
        metric_dict = {
            'val_loss': global_mean_loss,
        }
        self.log_dict(
            metric_dict,
            prog_bar=True,
            # on_step=False,
            on_epoch=True,
            # sync_dist=True,
        )