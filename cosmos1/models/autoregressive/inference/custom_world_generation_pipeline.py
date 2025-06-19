from typing import List, Optional, Tuple, Union, Set

import gc
import os
import time
import numpy as np
import torch
from safetensors.torch import load_file
from einops import rearrange
from omegaconf import DictConfig, ListConfig

from cosmos1.models.autoregressive.configs.base.tokenizer import TokenizerConfig
from cosmos1.models.autoregressive.configs.inference.inference_config import (
    DataShapeConfig,
    DiffusionDecoderSamplingConfig,
    InferenceConfig,
    SamplingConfig,
)
from cosmos1.models.autoregressive.diffusion_decoder.inference import diffusion_decoder_process_tokens
from cosmos1.models.autoregressive.diffusion_decoder.model import LatentDiffusionDecoderModel
from cosmos1.models.autoregressive.model import AutoRegressiveModel
from cosmos1.utils import log
from cosmos1.models.common.base_world_generation_pipeline import BaseWorldGenerationPipeline
from cosmos1.models.autoregressive.model import AutoRegressiveModel
from cosmos1.models.autoregressive.tokenizer.tokenizer import DiscreteMultimodalTokenizer, update_vocab_size
from cosmos1.models.autoregressive.inference.world_generation_pipeline import create_inference_config, detect_model_size_from_ckpt_path
from cosmos1.models.autoregressive.utils.inference import prepare_array_batch
from cosmos1.models.diffusion.inference.inference_utils import (
    load_model_by_config,
    load_network_model,
    load_tokenizer_model,
)
from cosmos1.models.autoregressive.utils.checkpoint import process_state_dict
from cosmos1.models.autoregressive.utils.sampling import decode_n_tokens, decode_one_token, prefill
from cosmos1.utils import log, misc
from cosmo_lightning.models.universal_multi_decoder_vae_module import UniversalMultiDecoderVAEModule


class CustomARBaseGenerationPipeline(BaseWorldGenerationPipeline):
    def __init__(
        self,
        inference_type: str,
        checkpoint_dir: str,
        checkpoint_name: str,
        tokenizer_path: str,
        # additional parameters
        dataset_name: str,
        channel_names: List[str],
        omega_conf: Union[DictConfig, ListConfig, None],
        # max_latent_shape: List[int],
        # dataset_latent_shape: List[int],
        #########################
        enable_text_guardrail: bool = False,
        enable_video_guardrail: bool = True,
        offload_network: bool = False,
        offload_tokenizer: bool = False,
        disable_diffusion_decoder: bool = False,
        offload_guardrail_models: bool = False,
        offload_diffusion_decoder: bool = False,
        overwrite_ckpt_path: str = None,
        video_height: int = 1024,
        video_width: int = 640,
        context_len: int = 9,
        compression_ratio: List[int] = [8, 16, 16],
        ignore_first_latent: bool = False,
    ):
        assert inference_type in [
            "base",
            "video2world",
        ], "Invalid inference_type, must be 'base' or 'video2world'"

        # Create inference config
        if overwrite_ckpt_path is not None:
            model_size = detect_model_size_from_ckpt_path(overwrite_ckpt_path)
        else:
            model_size = detect_model_size_from_ckpt_path(checkpoint_name)
        model_ckpt_path = os.path.join(checkpoint_dir, checkpoint_name, "model.pt")
        tokenizer_ckpt_path = tokenizer_path

        inference_config: InferenceConfig = create_inference_config(
            model_ckpt_path=model_ckpt_path,
            tokenizer_ckpt_path=tokenizer_ckpt_path,
            model_size=model_size,
            inference_type=inference_type,
            overwrite_ckpt_path=overwrite_ckpt_path,
            video_height=video_height,
            video_width=video_width,
            compression_ratio=compression_ratio,
            ignore_first_latent=ignore_first_latent,
        )
        
        # update self.inference_config to include dataset_name and channel_names
        inference_config.tokenizer_config.video_tokenizer.dataset_name = dataset_name
        inference_config.tokenizer_config.video_tokenizer.channel_names = channel_names
        inference_config.tokenizer_config.video_tokenizer.omega_conf = omega_conf
        # inference_config.tokenizer_config.video_tokenizer.max_latent_shape = max_latent_shape
        # inference_config.tokenizer_config.video_tokenizer.dataset_latent_shape = dataset_latent_shape

        self.inference_config = inference_config
        self.disable_diffusion_decoder = disable_diffusion_decoder

        if not disable_diffusion_decoder:
            self.diffusion_decoder_ckpt_path = os.path.join(
                checkpoint_dir, "Cosmos-1.0-Diffusion-7B-Decoder-DV8x16x16ToCV8x8x8/model.pt"
            )
            self.diffusion_decoder_config = "DD_FT_7Bv1_003_002_tokenizer888_spatch2_discrete_cond_on_token"
            self.diffusion_decoder_tokenizer_path = os.path.join(checkpoint_dir, "Cosmos-1.0-Tokenizer-CV8x8x8")
            self.dd_sampling_config = DiffusionDecoderSamplingConfig()
            aux_vars_path = os.path.join(os.path.dirname(self.diffusion_decoder_ckpt_path), "aux_vars.pt")
            # We use a generic prompt when no text prompts are available for diffusion decoder.
            # Generic prompt used - "high quality, 4k, high definition, smooth video"
            aux_vars = torch.load(aux_vars_path, weights_only=True)
            self.generic_prompt = dict()
            self.generic_prompt["context"] = aux_vars["context"].cuda()
            self.generic_prompt["context_mask"] = aux_vars["context_mask"].cuda()

        self.latent_shape = inference_config.data_shape_config.latent_shape  # [L, 40, 64]
        self._supported_context_len = [context_len]
        self.tokenizer_config = inference_config.tokenizer_config

        self.offload_diffusion_decoder = offload_diffusion_decoder
        self.diffusion_decoder_model = None
        if not self.offload_diffusion_decoder and not disable_diffusion_decoder:
            self._load_diffusion_decoder()

        super().__init__(
            inference_type=inference_type,
            checkpoint_dir=checkpoint_dir,
            checkpoint_name=checkpoint_name,
            enable_text_guardrail=enable_text_guardrail,
            enable_video_guardrail=enable_video_guardrail,
            offload_guardrail_models=offload_guardrail_models,
            offload_network=offload_network,
            offload_tokenizer=offload_tokenizer,
            offload_text_encoder_model=True,
        )
    
    def _load_model(self):
        """Load and initialize the autoregressive model.

        Creates and configures the autoregressive model with appropriate settings.
        """
        self.model = CustomAutoRegressiveModel(
            config=self.inference_config.model_config,
        )
    
    def _load_network(self):
        """Load network weights for the autoregressive model."""
        self.model.load_ar_model(tokenizer_config=self.inference_config.tokenizer_config)

    def _load_tokenizer(self):
        """Load and initialize the tokenizer model.

        Configures the tokenizer using settings from inference_config and
        attaches it to the autoregressive model.
        """
        self.model.load_tokenizer(tokenizer_config=self.inference_config.tokenizer_config)

    def _load_diffusion_decoder(self):
        """Load and initialize the diffusion decoder model."""
        self.diffusion_decoder_model = load_model_by_config(
            config_job_name=self.diffusion_decoder_config,
            config_file="cosmos1/models/autoregressive/diffusion_decoder/config/config_latent_diffusion_decoder.py",
            model_class=LatentDiffusionDecoderModel,
        )
        load_network_model(self.diffusion_decoder_model, self.diffusion_decoder_ckpt_path)
        load_tokenizer_model(self.diffusion_decoder_model, self.diffusion_decoder_tokenizer_path)

    def _offload_diffusion_decoder(self):
        """Offload diffusion decoder model from GPU memory."""
        if self.diffusion_decoder_model is not None:
            del self.diffusion_decoder_model
            self.diffusion_decoder_model = None
        gc.collect()
        torch.cuda.empty_cache()

    def _run_model_with_offload(
        self, inp_vid: torch.Tensor, num_input_frames: int, seed: int, sampling_config: SamplingConfig
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Run the autoregressive model to generate video tokens.

        Takes input video frames and generates new video tokens using the autoregressive model.
        Handles context frame selection and token generation.

        Args:
            inp_vid (torch.Tensor): Input video tensor of shape
            num_input_frames (int): Number of context frames to use from input. The tensor shape should be (B x T x 3 x H x W).
            seed (int): Random seed for generation
            sampling_config (SamplingConfig): Configuration for sampling parameters

        Returns:
            tuple: (
                List of generated video tensors,
                List of token index tensors,
                List of prompt embedding tensors
            )
        """
        # Choosing the context length from list of available contexts
        latent_context_t_size = 0
        context_used = 0
        for _clen in self._supported_context_len:
            if num_input_frames >= _clen:
                context_used = _clen
                latent_context_t_size += 1
        latent_context_t_size = 2
        log.info(f"Using input size of {context_used} frames with latent context size of {latent_context_t_size}")

        data_batch = {"video": inp_vid}
        data_batch = misc.to(data_batch, "cuda")

        T, H, W = self.latent_shape
        num_gen_tokens = int(np.prod([T - latent_context_t_size, H, W]))

        out_videos_cur_batch, indices_tensor_cur_batch = self.generate_partial_tokens_from_data_batch(
            data_batch=data_batch,
            num_tokens_to_generate=num_gen_tokens,
            sampling_config=sampling_config,
            tokenizer_config=self.tokenizer_config,
            latent_shape=self.latent_shape,
            task_condition="video",
            num_chunks_to_generate=1,
            seed=seed,
        )
        if self.offload_network:
            self._offload_network()
        if self.offload_tokenizer:
            self._offload_tokenizer()
        return out_videos_cur_batch, indices_tensor_cur_batch

    def _run_diffusion_decoder(
        self,
        out_videos_cur_batch: List[torch.Tensor],
        indices_tensor_cur_batch: List[torch.Tensor],
        t5_emb_batch: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """Process generated tokens through the diffusion decoder.

        Enhances video quality through diffusion-based decoding.

        Args:
            out_videos_cur_batch: List of generated video tensors
            indices_tensor_cur_batch: List of token indices tensors
            t5_emb_batch: List of text embeddings for conditioning

        Returns:
            list: Enhanced video tensors after diffusion processing
        """
        out_videos_cur_batch_dd = diffusion_decoder_process_tokens(
            model=self.diffusion_decoder_model,
            indices_tensor=indices_tensor_cur_batch,
            dd_sampling_config=self.dd_sampling_config,
            original_video_example=out_videos_cur_batch[0],
            t5_emb_batch=t5_emb_batch,
        )
        return out_videos_cur_batch_dd

    def _run_diffusion_decoder_with_offload(
        self,
        out_videos_cur_batch: List[torch.Tensor],
        indices_tensor_cur_batch: List[torch.Tensor],
        t5_emb_batch: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """Run diffusion decoder with memory management.

        Loads decoder if needed, processes videos, and offloads decoder afterward
        if configured in offload_diffusion_decoder.

        Args:
            out_videos_cur_batch: List of generated video tensors
            indices_tensor_cur_batch: List of token indices tensors
            t5_emb_batch: List of text embeddings for conditioning

        Returns:
            list: Enhanced video tensors after diffusion processing
        """
        if self.offload_diffusion_decoder:
            self._load_diffusion_decoder()
        out_videos_cur_batch = self._run_diffusion_decoder(out_videos_cur_batch, indices_tensor_cur_batch, t5_emb_batch)
        if self.offload_diffusion_decoder:
            self._offload_diffusion_decoder()
        return out_videos_cur_batch

    def generate(
        self,
        inp_vid: torch.Tensor,
        sampling_config: SamplingConfig,
        num_input_frames: int = 9,
        seed: int = 0,
    ) -> np.ndarray | None:
        """Generate a video continuation from input frames.

        Pipeline steps:
        1. Generates video tokens using autoregressive model
        2. Optionally enhances quality via diffusion decoder
        3. Applies safety checks if enabled

        Args:
            inp_vid: Input video tensor of shape (batch_size, time, channels=3, height, width)
            sampling_config: Parameters controlling the generation process
            num_input_frames: Number of input frames to use as context (default: 9)
            seed: Random seed for reproducibility (default: 0)

        Returns:
            np.ndarray | None: Generated video as numpy array (time, height, width, channels)
                if generation successful, None if safety checks fail
        """
        log.info("Run generation")
        out_videos_cur_batch, indices_tensor_cur_batch = self._run_model_with_offload(
            inp_vid, num_input_frames, seed, sampling_config
        )
        log.info("Finish AR model generation")

        if not self.disable_diffusion_decoder:
            log.info("Run diffusion decoder on generated tokens")
            out_videos_cur_batch = self._run_diffusion_decoder_with_offload(
                out_videos_cur_batch, indices_tensor_cur_batch, t5_emb_batch=[self.generic_prompt["context"]]
            )
            log.info("Finish diffusion decoder on generated tokens")
        out_videos_cur_batch = prepare_array_batch(out_videos_cur_batch)
        output_video = out_videos_cur_batch[0]

        # if self.enable_video_guardrail:
        #     log.info("Run guardrail on generated video")
        #     output_video = self._run_guardrail_on_video_with_offload(output_video)
        #     if output_video is None:
        #         log.critical("Generated video is not safe")
        #         return None
        #     log.info("Finish guardrail on generated video")

        return output_video

    @torch.inference_mode()
    def generate_partial_tokens_from_data_batch(
        self,
        data_batch: dict,
        num_tokens_to_generate: int,
        sampling_config: SamplingConfig,
        tokenizer_config: TokenizerConfig,
        latent_shape: list[int],
        task_condition: str,
        num_chunks_to_generate: int = 1,
        seed: int = 0,
    ) -> tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """Generate video tokens from partial input tokens with conditioning.

        Handles token generation and decoding process:
        1. Processes input batch and applies conditioning
        2. Generates specified number of new tokens
        3. Decodes tokens to video frames

        Args:
            data_batch: Dictionary containing input data including video and optional context
            num_tokens_to_generate: Number of tokens to generate
            sampling_config: Configuration for sampling parameters
            tokenizer_config: Configuration for tokenizer, including video tokenizer settings
            latent_shape: Shape of video latents [T, H, W]
            task_condition: Type of generation task ('video' or 'text_and_video')
            num_chunks_to_generate: Number of chunks to generate (default: 1)
            seed: Random seed for generation (default: 0)

        Returns:
            tuple containing:
                - List[torch.Tensor]: Generated videos
                - List[torch.Tensor]: Input videos
                - List[torch.Tensor]: Generated tokens
                - List[torch.Tensor]: Token index tensors
        """
        log.debug(f"Starting generate_partial_tokens_from_data_batch with seed {seed}")
        log.debug(f"Number of tokens to generate: {num_tokens_to_generate}")
        log.debug(f"Latent shape: {latent_shape}")

        video_token_start = tokenizer_config.video_tokenizer.tokenizer_offset
        video_vocab_size = tokenizer_config.video_tokenizer.vocab_size
        video_token_end = video_token_start + video_vocab_size

        logit_clipping_range = [video_token_start, video_token_end]

        if self.offload_network:
            self._offload_network()
        if self.offload_tokenizer:
            self._load_tokenizer()

        assert logit_clipping_range == [
            0,
            self.model.tokenizer.video_vocab_size,
        ], f"logit_clipping_range {logit_clipping_range} is not supported for fast generate. Expected [0, {self.model.tokenizer.video_vocab_size}]"

        out_videos = {}
        out_indices_tensors = {}

        # for text2world, we only add a <bov> token at the beginning of the video tokens, this applies to 5B and 13B models
        if self.model.tokenizer.tokenizer_config.training_type == "text_to_video":
            num_bov_tokens = 1
            num_eov_tokens = 0
        else:
            num_eov_tokens = 1 if self.model.tokenizer.tokenizer_config.add_special_tokens else 0
            num_bov_tokens = 1 if self.model.tokenizer.tokenizer_config.add_special_tokens else 0

        chunk_idx = 0
        out_videos[chunk_idx] = []
        out_indices_tensors[chunk_idx] = []

        # get the context embedding and mask
        context = data_batch.get("context", None) if task_condition != "video" else None
        context_mask = data_batch.get("context_mask", None) if task_condition != "video" else None
        if context is not None:
            context = misc.to(context, "cuda").detach().clone()
        if context_mask is not None:
            context_mask = misc.to(context_mask, "cuda").detach().clone()

        # get the video tokens
        data_tokens, token_boundaries = self.model.tokenizer.tokenize(data_batch=data_batch)
        data_tokens = misc.to(data_tokens, "cuda").detach().clone()
        batch_size = data_tokens.shape[0]

        for sample_num in range(batch_size):
            input_tokens = data_tokens[sample_num][0 : token_boundaries["video"][sample_num][1]]  # [B, L]
            input_tokens = [
                input_tokens[0 : -num_tokens_to_generate - num_eov_tokens].tolist()
            ]  # -1 is to exclude eov token
            log.debug(
                f"Run sampling. # input condition tokens: {len(input_tokens[0])}; # generate tokens: {num_tokens_to_generate + num_eov_tokens}; "
                f"full length of the data tokens: {len(data_tokens[sample_num])}: {data_tokens[sample_num]}"
            )
            video_start_boundary = token_boundaries["video"][sample_num][0] + num_bov_tokens

            video_decoded, indices_tensor = self.generate_video_from_tokens(
                prompt_tokens=input_tokens,
                latent_shape=latent_shape,
                video_start_boundary=video_start_boundary,
                max_gen_len=num_tokens_to_generate,
                sampling_config=sampling_config,
                logit_clipping_range=logit_clipping_range,
                seed=seed,
                context=context,
                context_mask=context_mask,
            )  # BCLHW, range [0, 1]

            # For the first chunk, we store the entire generated video
            out_videos[chunk_idx].append(video_decoded[sample_num].detach().clone())
            out_indices_tensors[chunk_idx].append(indices_tensor[sample_num].detach().clone())

        output_videos = []
        output_indice_tensors = []
        for sample_num in range(len(out_videos[0])):
            tensors_to_concat = [out_videos[chunk_idx][sample_num] for chunk_idx in range(num_chunks_to_generate)]
            concatenated = torch.cat(tensors_to_concat, dim=1)
            output_videos.append(concatenated)

            indices_tensor_to_concat = [
                out_indices_tensors[chunk_idx][sample_num] for chunk_idx in range(num_chunks_to_generate)
            ]
            concatenated_indices_tensor = torch.cat(indices_tensor_to_concat, dim=1)  # BLHW
            output_indice_tensors.append(concatenated_indices_tensor)

        return output_videos, output_indice_tensors
    
    def generate_video_from_tokens(
        self,
        prompt_tokens: list[torch.Tensor],
        latent_shape: list[int],
        video_start_boundary: int,
        max_gen_len: int,
        sampling_config: SamplingConfig,
        logit_clipping_range: list[int],
        seed: int = 0,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Combine the tokens and do padding, sometimes the generated tokens end before the max_gen_len
        total_seq_len = np.prod(latent_shape)

        assert not sampling_config.logprobs

        stop_tokens = self.model.tokenizer.stop_tokens
        if self.offload_tokenizer:
            self._offload_tokenizer()
        if self.offload_network:
            self._load_network()

        generation_tokens, _ = self.model.generate(
            prompt_tokens=prompt_tokens,
            temperature=sampling_config.temperature,
            top_p=sampling_config.top_p,
            echo=sampling_config.echo,
            seed=seed,
            context=context,
            context_mask=context_mask,
            max_gen_len=max_gen_len,
            compile_sampling=sampling_config.compile_sampling,
            compile_prefill=sampling_config.compile_prefill,
            stop_tokens=stop_tokens,
            verbose=True,
        )
        generation_tokens = generation_tokens[:, video_start_boundary:]
        # Combine the tokens and do padding, sometimes the generated tokens end before the max_gen_len
        if generation_tokens.shape[1] < total_seq_len:
            log.warning(
                f"Generated video tokens (shape:{generation_tokens.shape}) shorted than expected {total_seq_len}. Could be the model produce end token early. Repeat the last token to fill the sequence in order for decoding."
            )
            padding_len = total_seq_len - generation_tokens.shape[1]
            padding_tokens = generation_tokens[:, [-1]].repeat(1, padding_len)
            generation_tokens = torch.cat([generation_tokens, padding_tokens], dim=1)
        # Cast to LongTensor
        indices_tensor = generation_tokens.long()
        # First, we reshape the generated tokens into batch x time x height x width
        indices_tensor = rearrange(
            indices_tensor,
            "B (T H W) -> B T H W",
            T=latent_shape[0],
            H=latent_shape[1],
            W=latent_shape[2],
        )
        log.debug(f"generated video tokens {len(generation_tokens[0])} -> reshape: {indices_tensor.shape}")
        # If logit clipping range is specified, offset the generated indices by the logit_clipping_range[0]
        # Video decoder always takes tokens in the range (0, N-1). So, this offset is needed.
        if len(logit_clipping_range) > 0:
            indices_tensor = indices_tensor - logit_clipping_range[0]

        if self.offload_network:
            self._offload_network()
        if self.offload_tokenizer:
            self._load_tokenizer()

        # Now decode the video using tokenizer.
        quant_codes = self.model.tokenizer.video_tokenizer.quantizer.indices_to_codes(indices_tensor)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            video_decoded = self.model.tokenizer.video_tokenizer.decode(quant_codes, self.tokenizer_config.video_tokenizer.dataset_name)
            var_ids = self.model.tokenizer.video_tokenizer.encoder.patcher3d.get_var_ids(tuple(self.inference_config.tokenizer_config.video_tokenizer.channel_names), video_decoded.device)
            video_decoded = video_decoded[:, var_ids]
        # Normalize decoded video from [-1, 1] to [0, 1], and clip value
        # video_decoded = (video_decoded * 0.5 + 0.5).clamp_(0, 1)
        return video_decoded, indices_tensor


class CustomAutoRegressiveModel(AutoRegressiveModel):
    def load_tokenizer(self, tokenizer_config):
        """
        Load the tokenizer.
        """
        self.tokenizer = CustomDiscreteMultimodalTokenizer(tokenizer_config)


class CustomDiscreteMultimodalTokenizer(DiscreteMultimodalTokenizer):
    def _build_video_tokenizer(self):
        r"""Function to initialize the video tokenizer model."""
        if self.tokenizer_config.video_tokenizer is not None:
            lightning_module = UniversalMultiDecoderVAEModule(**self.tokenizer_config.video_tokenizer.omega_conf['model'])
            ckpt = torch.load(self.tokenizer_config.video_tokenizer.tokenizer_ckpt_path, map_location='cpu')
            state_dict = ckpt['state_dict']
            msg = lightning_module.load_state_dict(state_dict, strict=True)
            print(f"Loaded tokenizer from {self.tokenizer_config.video_tokenizer.tokenizer_ckpt_path}: {msg}")
            self.video_tokenizer = lightning_module.model
            self.video_tokenizer.to("cuda").eval()
            
            self.video_vocab_size = self.tokenizer_config.video_tokenizer.vocab_size
            special_token_offset = (
                self.tokenizer_config.video_tokenizer.tokenizer_offset
                + self.tokenizer_config.video_tokenizer.vocab_size
            )
            self.video_special_tokens = {
                "<|begin_of_video|>": special_token_offset,
                "<|end_of_video|>": special_token_offset + 1,
                "<|pad_token_video|>": special_token_offset + 2,
            }

            self.vocab_size = update_vocab_size(
                existing_vocab_size=self.vocab_size,
                to_be_added_vocab_size=self.tokenizer_config.video_tokenizer.vocab_size,
                training_type=self.training_type,
                add_special_tokens=self.tokenizer_config.add_special_tokens,
                video_special_tokens=self.video_special_tokens,
            )
        else:
            self.video_tokenizer = None
    
    def _tokenize_video(self, videos: torch.Tensor, pixel_chunk_duration: Optional[int] = None):
        r"""Function to tokenize video.
        Args:
            videos (torch.Tensor): Input video data tensor
            pixel_chunk_duration (Optional[float]): Pixel chunk duration. If provided, we pass it to the video tokenizer.
        Returns:
            video_tokens (list[list[int]]): List of video tokens
        """

        video_tokens = []
        batch_size = videos.shape[0]

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            (quant_info, _, _), _ = self.video_tokenizer.encode(
                videos,
                dataset_name=self.tokenizer_config.video_tokenizer.dataset_name,
                variables=self.tokenizer_config.video_tokenizer.channel_names
            )

        # Flatten the indices
        indices = rearrange(quant_info, "B T H W -> B (T H W)")

        # tokenizer_offset tells what offset should be added to the tokens.
        # This is needed for vocab expansion.
        indices += self.tokenizer_config.video_tokenizer.tokenizer_offset

        # Add begin and end of video tokens
        bov_token = self.video_special_tokens["<|begin_of_video|>"]
        eov_token = self.video_special_tokens["<|end_of_video|>"]

        # Append bov and eov tokens
        if self.tokenizer_config.add_special_tokens:
            for i in range(batch_size):
                video_tokens.append([bov_token] + indices[i].tolist() + [eov_token])
        else:
            if self.training_type == "text_to_video":
                for i in range(batch_size):
                    video_tokens.append([bov_token] + indices[i].tolist())
            else:
                for i in range(batch_size):
                    video_tokens.append(indices[i].tolist())
                    assert (
                        len(video_tokens[-1]) == self.tokenizer_config.video_tokenizer.max_seq_len
                    ), f"Expected {self.tokenizer_config.video_tokenizer.max_seq_len} tokens, got {len(video_tokens[-1])}; video shape: {videos.shape}"

        return video_tokens