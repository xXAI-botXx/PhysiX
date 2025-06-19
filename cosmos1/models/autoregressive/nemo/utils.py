# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gc
import importlib
import math
import os
from typing import List
import h5py

import torch
import torchvision
from huggingface_hub import snapshot_download

from cosmos1.models.autoregressive.configs.inference.inference_config import DiffusionDecoderSamplingConfig
from cosmos1.models.autoregressive.diffusion_decoder.inference import diffusion_decoder_process_tokens
from cosmos1.models.autoregressive.diffusion_decoder.model import LatentDiffusionDecoderModel
# from cosmos1.models.diffusion.inference.inference_utils import (
#     load_network_model,
#     load_tokenizer_model,
#     skip_init_linear,
# )
from cosmos1.utils import log
from cosmos1.utils.config_helper import get_config_module, override

# TOKENIZER_COMPRESSION_FACTOR = [8, 16, 16]
# DATA_RESOLUTION_SUPPORTED = [640, 1024]
NUM_CONTEXT_FRAMES = int(os.environ["COSMOS_AR_NUM_TOTAL_FRAMES"])


def resize_input(video: torch.Tensor, dimensions: List[int]):
    r"""
    Function to perform aspect ratio preserving resizing and center cropping.
    This is needed to make the video into target resolution.
    Args:
        video (torch.Tensor): Input video tensor
        resolution (list[int]): Data resolution
    Returns:
        Cropped video
    """

    orig_h, orig_w = video.shape[2], video.shape[3]
    target_h, target_w = dimensions
    
    if orig_h == target_h and orig_w == target_w:
        return video

    scaling_ratio = max((target_w / orig_w), (target_h / orig_h))
    resizing_shape = (int(math.ceil(scaling_ratio * orig_h)), int(math.ceil(scaling_ratio * orig_w)))
    video_resized = torchvision.transforms.functional.resize(video, resizing_shape)
    video_cropped = torchvision.transforms.functional.center_crop(video_resized, dimensions)
    return video_cropped


def _prepare_video_tensor(
    video: torch.Tensor,
    dimensions: List[int],
    sliding_windows: bool = False
) -> torch.Tensor:
    """Common preprocessing steps for video tensors.
    
    Args:
        video (torch.Tensor): Input video tensor of shape [frames, height, width, channels]
        dimensions (List[int]): Target spatial dimensions [height, width]
        sliding_windows (bool): If True, return all sliding windows of size NUM_CONTEXT_FRAMES. Default: False.
        
    Returns:
        torch.Tensor: Processed video tensor. If sliding_windows is True, shape is [num_windows, channels, NUM_CONTEXT_FRAMES, height, width]. Otherwise, [1, channels, NUM_CONTEXT_FRAMES, height, width].
    """
    n_frames = video.shape[0]
    if sliding_windows and n_frames >= NUM_CONTEXT_FRAMES:
        return _generate_sliding_windows(video, dimensions)
    else:
        return _process_single_window(video, dimensions)


def _generate_sliding_windows(video: torch.Tensor, dimensions: List[int]) -> torch.Tensor:
    """Generate and process all sliding windows of size NUM_CONTEXT_FRAMES."""
    n_frames = video.shape[0]
    num_windows = n_frames - NUM_CONTEXT_FRAMES + 1
    # Generate sliding windows: [num_windows, NUM_CONTEXT_FRAMES, H, W, C]
    windows = video.unfold(0, NUM_CONTEXT_FRAMES, 1)
    # return windows.permute(0, 3, 4, 1, 2)[:100:20]
    # Permute to [num_windows, NUM_CONTEXT_FRAMES, C, H, W]
    # random selection of windows
    windows = windows.permute(0, 4, 3, 1, 2)#[torch.randperm(num_windows)[:8]]
    # Combine windows and frames for batch resizing
    combined = windows.reshape(-1, windows.size(2), windows.size(3), windows.size(4))
    resized = resize_input(combined, dimensions)
    # Reshape back and transpose dimensions
    resized = resized.reshape(num_windows, NUM_CONTEXT_FRAMES, resized.size(1), dimensions[0], dimensions[1])
    windows = resized.transpose(1, 2)  # [num_windows, C, NUM_CONTEXT_FRAMES, H, W]
    return windows


def _process_single_window(video: torch.Tensor, dimensions: List[int]) -> torch.Tensor:
    """Process a single window, handling truncation/padding as needed."""
    n_frames = video.shape[0]
    # Truncate or pad video to NUM_CONTEXT_FRAMES
    if n_frames > NUM_CONTEXT_FRAMES:
        video = video[:NUM_CONTEXT_FRAMES, :, :, :]
    elif n_frames < NUM_CONTEXT_FRAMES:
        log.info(f"Video doesn't have {NUM_CONTEXT_FRAMES} frames. Padding with last frame.")
        padding = video[-1].unsqueeze(0).repeat(NUM_CONTEXT_FRAMES - n_frames, 1, 1, 1)
        video = torch.cat([video, padding], dim=0)
    # Process frames: [NUM_CONTEXT_FRAMES, H, W, C] -> [1, C, NUM_CONTEXT_FRAMES, H, W]
    video = video.permute(0, 3, 1, 2)  # [NUM_CONTEXT_FRAMES, C, H, W]
    video = resize_input(video, dimensions)  # [NUM_CONTEXT_FRAMES, C, H', W']
    video = video.transpose(0, 1).unsqueeze(0)  # Add batch dimension
    return video


def read_input_videos(input_video: str, dimensions: List[int]) -> torch.Tensor:
    """Read and process an MP4 video file.

    Args:
        input_video (str): Path to .mp4 file

    Returns:
        torch.Tensor: Processed video tensor
    """
    video, _, _ = torchvision.io.read_video(input_video)
    video = video.float() / 255.0
    video = video * 2 - 1
    return _prepare_video_tensor(video, dimensions)


def read_input_array(input_array: str, dimensions: List[int], sliding_windows: bool = False) -> torch.Tensor:
    """Read and process an HDF5 array file.

    Args:
        input_array (str): Path to HDF5 file

    Returns:
        torch.Tensor: Processed video tensor
    """
    array = h5py.File(input_array, "r")['data'][:]
    video = torch.from_numpy(array)
    video = video.float()
    return _prepare_video_tensor(video, dimensions, sliding_windows=sliding_windows)


def run_diffusion_decoder_model(indices_tensor_cur_batch: List[torch.Tensor], out_videos_cur_batch):
    """Run a 7b diffusion model to enhance generation output

    Args:
        indices_tensor_cur_batch (List[torch.Tensor]): The index tensor(i.e) prompt + generation tokens
        out_videos_cur_batch (torch.Tensor): The output decoded video of shape [bs, 3, 33, 640, 1024]
    """
    diffusion_decoder_ckpt_path = snapshot_download("nvidia/Cosmos-1.0-Diffusion-7B-Decoder-DV8x16x16ToCV8x8x8")
    dd_tokenizer_dir = snapshot_download("nvidia/Cosmos-1.0-Tokenizer-CV8x8x8")
    tokenizer_corruptor_dir = snapshot_download("nvidia/Cosmos-1.0-Tokenizer-DV8x16x16")

    diffusion_decoder_model = load_model_by_config(
        config_job_name="DD_FT_7Bv1_003_002_tokenizer888_spatch2_discrete_cond_on_token",
        config_file="cosmos1/models/autoregressive/diffusion_decoder/config/config_latent_diffusion_decoder.py",
        model_class=LatentDiffusionDecoderModel,
        encoder_path=os.path.join(tokenizer_corruptor_dir, "encoder.jit"),
        decoder_path=os.path.join(tokenizer_corruptor_dir, "decoder.jit"),
    )
    load_network_model(diffusion_decoder_model, os.path.join(diffusion_decoder_ckpt_path, "model.pt"))
    load_tokenizer_model(diffusion_decoder_model, dd_tokenizer_dir)

    generic_prompt = dict()
    aux_vars = torch.load(os.path.join(diffusion_decoder_ckpt_path, "aux_vars.pt"), weights_only=True)
    generic_prompt["context"] = aux_vars["context"].cuda()
    generic_prompt["context_mask"] = aux_vars["context_mask"].cuda()

    output_video = diffusion_decoder_process_tokens(
        model=diffusion_decoder_model,
        indices_tensor=indices_tensor_cur_batch,
        dd_sampling_config=DiffusionDecoderSamplingConfig(),
        original_video_example=out_videos_cur_batch[0],
        t5_emb_batch=[generic_prompt["context"]],
    )

    del diffusion_decoder_model
    diffusion_decoder_model = None
    gc.collect()
    torch.cuda.empty_cache()

    return output_video


def load_model_by_config(
    config_job_name,
    config_file="projects/cosmos_video/config/config.py",
    model_class=LatentDiffusionDecoderModel,
    encoder_path=None,
    decoder_path=None,
):
    config_module = get_config_module(config_file)
    config = importlib.import_module(config_module).make_config()

    config = override(config, ["--", f"experiment={config_job_name}"])

    # Check that the config is valid
    config.validate()
    # Freeze the config so developers don't change it during training.
    config.freeze()  # type: ignore
    if encoder_path:
        config.model.tokenizer_corruptor["enc_fp"] = encoder_path
    if decoder_path:
        config.model.tokenizer_corruptor["dec_fp"] = decoder_path
    # Initialize model
    with skip_init_linear():
        model = model_class(config.model)
    return model
