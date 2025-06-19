import argparse
import json
import math
import os
import logging
from pathlib import Path
from typing import List, Dict, Optional
import h5py

import numpy as np
import torch
import torchvision
from PIL import Image

from cosmos1.models.autoregressive.configs.inference.inference_config import SamplingConfig
from cosmos1.utils import log
import  os

# Constants
_IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".webp"]
_VIDEO_EXTENSIONS = [".mp4"]
_SUPPORTED_CONTEXT_LEN = [1, 9]
NUM_TOTAL_FRAMES = int(os.environ.get('COSMOS_AR_NUM_TOTAL_FRAMES',13))
NORMALIZE_TENSOR = False

###########################
# Core Processing Functions
###########################

def resize_input(video: torch.Tensor, resolution: list[int]) -> torch.Tensor:
    """Perform aspect ratio preserving resizing and center cropping."""
    orig_h, orig_w = video.shape[2], video.shape[3]
    target_h, target_w = resolution

    scaling_ratio = max((target_w / orig_w), (target_h / orig_h))
    resizing_shape = (int(math.ceil(scaling_ratio * orig_h)), 
                     int(math.ceil(scaling_ratio * orig_w)))
    video_resized = torchvision.transforms.functional.resize(video, resizing_shape)
    return torchvision.transforms.functional.center_crop(video_resized, resolution)

###########################
# Helper Functions
###########################

def _normalize_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Normalize tensor from [0,1] to [-1,1] range."""
    return tensor * 2 - 1

def _process_frames(video: torch.Tensor, num_input: int, use_first: bool) -> Optional[torch.Tensor]:
    """Select and pad frames according to configuration."""
    if video.shape[0] < num_input:
        return None
    
    selected = video[:num_input] if use_first else video[-num_input:]
    padding = selected[-1].unsqueeze(0).repeat(NUM_TOTAL_FRAMES - num_input, 1, 1, 1)
    return torch.cat([selected, padding], dim=0)

def _prepare_output(video: torch.Tensor, resolution: List[int]) -> torch.Tensor:
    """Final processing steps for output preparation."""
    video = video.permute(0, 3, 1, 2)  # THWC -> TCHW
    video = resize_input(video, resolution)
    return video.transpose(0, 1).unsqueeze(0)  # TCHW -> CTHW + batch dim

###########################
# Core Loader Functions
###########################

def load_array_from_list(flist: List[str], data_resolution: List[int], 
                        num_input_frames: int, use_first_frames: bool = False) -> dict:
    """Load array data from HDF5 files."""
    all_videos = {}
    for path in flist:
        ext = os.path.splitext(path)[-1].lower()
        if ext not in {".h5", ".hdf5"}:
            continue

        try:
            with h5py.File(path, "r") as hf:
                arr_data = hf["data"][:]
            if NORMALIZE_TENSOR:    
                video = torch.from_numpy(arr_data).float() / 255.0
                video = _normalize_tensor(video)
            else:
                video = torch.from_numpy(arr_data).float()
            
            processed = _process_frames(video, num_input_frames, use_first_frames)
            if processed is None:
                log.warning(f"Skipping {os.path.basename(path)}: insufficient frames")
                continue
                
            all_videos[os.path.basename(path)] = _prepare_output(processed, data_resolution)
        except Exception as e:
            log.error(f"Error processing {path}: {str(e)}")
    return all_videos

def load_videos_from_list(flist: List[str], data_resolution: List[int], 
                         num_input_frames: int, use_first_frames: bool = False) -> dict:
    """Load video files from list."""
    all_videos = {}
    for path in flist:
        ext = os.path.splitext(path)[-1].lower()
        if ext not in _VIDEO_EXTENSIONS:
            continue

        try:
            video, _, _ = torchvision.io.read_video(path, pts_unit="sec")
            if NORMALIZE_TENSOR:
                video = video.float() / 255.0
                video = _normalize_tensor(video)
            else:
                video = video.float()
            
            processed = _process_frames(video, num_input_frames, use_first_frames)
            if processed is None:
                log.warning(f"Skipping {os.path.basename(path)}: insufficient frames")
                continue
                
            all_videos[os.path.basename(path)] = _prepare_output(processed, data_resolution)
        except Exception as e:
            log.error(f"Error processing {path}: {str(e)}")
    return all_videos

def load_image_from_list(flist: List[str], data_resolution: List[int]) -> dict:
    """Load images from list."""
    all_videos = {}
    for path in flist:
        ext = os.path.splitext(path)[-1].lower()
        if ext not in _IMAGE_EXTENSIONS:
            continue

        try:
            img = Image.open(path)
            tensor = torchvision.transforms.functional.to_tensor(img)
            static_vid = tensor.unsqueeze(0).repeat(NUM_TOTAL_FRAMES, 1, 1, 1)
            static_vid = _normalize_tensor(static_vid)
            processed = _prepare_output(static_vid, data_resolution)
            all_videos[os.path.basename(path)] = processed
        except Exception as e:
            log.error(f"Error processing {path}: {str(e)}")
    return all_videos

###########################
# Input Reader Functions
###########################

def _read_input_generic(batch_path: Optional[str], single_path: Optional[str], loader, limit=10, **kwargs):
    """Generic input reader for batch/single path pattern."""
    flist = []
    if batch_path:
        if os.path.isdir(batch_path):
            # Recursively collect all files in the directory
            for root, _, files in os.walk(batch_path):
                for file in files:
                    flist.append(os.path.join(root, file))
        else:
            # Read from JSON file
            with open(batch_path, "r") as f:
                flist = [json.loads(line.strip())["visual_input"] for line in f]
        flist = flist[:limit]
    else:
        if single_path is not None:
            flist = [single_path]
    return loader(flist, **kwargs)

def read_input_arrays(batch_input_path: str, data_resolution: List[int], 
                     num_input_frames: int, use_first_frames: bool = False) -> dict:
    return _read_input_generic(batch_input_path, None, load_array_from_list,
                              data_resolution=data_resolution,
                              num_input_frames=num_input_frames,
                              use_first_frames=use_first_frames)

def read_input_array(input_path: str, data_resolution: List[int], 
                    num_input_frames: int, use_first_frames: bool = False) -> dict:
    return _read_input_generic(None, input_path, load_array_from_list,
                              data_resolution=data_resolution,
                              num_input_frames=num_input_frames,
                              use_first_frames=use_first_frames)

def read_input_videos(batch_input_path: str, data_resolution: List[int], 
                     num_input_frames: int, use_first_frames: bool = False) -> dict:
    return _read_input_generic(batch_input_path, None, load_videos_from_list,
                              data_resolution=data_resolution,
                              num_input_frames=num_input_frames,
                              use_first_frames=use_first_frames)

def read_input_video(input_path: str, data_resolution: List[int], 
                    num_input_frames: int, use_first_frames: bool = False) -> dict:
    return _read_input_generic(None, input_path, load_videos_from_list,
                              data_resolution=data_resolution,
                              num_input_frames=num_input_frames,
                              use_first_frames=use_first_frames)

def read_input_images(batch_input_path: str, data_resolution: List[int]) -> dict:
    return _read_input_generic(batch_input_path, None, load_image_from_list,
                              data_resolution=data_resolution)

def read_input_image(input_path: str, data_resolution: List[int]) -> dict:
    return _read_input_generic(None, input_path, load_image_from_list,
                              data_resolution=data_resolution)

###########################
# Remaining Original Functions
###########################

def add_common_arguments(parser):
    """Add common command line arguments."""
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", 
                       help="Base directory containing model checkpoints")
    parser.add_argument("--video_save_name", type=str, default="output",
                       help="Output filename for single video")
    parser.add_argument("--video_save_folder", type=str, default="outputs/",
                       help="Output folder for videos")
    parser.add_argument("--input_image_or_video_path", type=str,
                       help="Path to input image/video")
    parser.add_argument("--batch_input_path", type=str,
                       help="Path to a JSON file or directory containing batch inputs")
    parser.add_argument("--num_input_frames", type=int, default=9,
                    #    choices=_SUPPORTED_CONTEXT_LEN,
                       help="Input frames for world generation")
    parser.add_argument("--temperature", type=float, default=1.0,
                       help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.8,
                       help="Top-p sampling value")
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed")
    parser.add_argument("--disable_diffusion_decoder", action="store_true",
                       help="Disable diffusion decoder")
    parser.add_argument("--offload_guardrail_models", action="store_true",
                       help="Offload guardrail models")
    parser.add_argument("--offload_diffusion_decoder", action="store_true",
                       help="Offload diffusion decoder")
    parser.add_argument("--offload_ar_model", action="store_true",
                       help="Offload AR model")
    parser.add_argument("--offload_tokenizer", action="store_true",
                       help="Offload tokenizer")
    parser.add_argument("--use_first_frames", action="store_true",
                       help="Use first frames instead of last")

def validate_args(args: argparse.Namespace, inference_type: str):
    """Validate command line arguments."""
    assert inference_type in ["base", "video2world"], "Invalid inference type"
    
    if args.input_type in ["image", "text_and_image"] and args.num_input_frames != 1:
        args.num_input_frames = 1
        log.info(f"Set num_input_frames to 1 for {args.input_type} input")

    if args.num_input_frames == 1:
        model_size = os.path.basename(args.ar_model_dir)
        if "4B" in model_size:
            log.warning("4B model has ~15% failure rate with image input")
        elif "5B" in model_size:
            log.warning("5B model has ~7% failure rate with image input")

    assert (args.input_image_or_video_path or args.batch_input_path), "Missing input path"
    if inference_type == "video2world" and not args.batch_input_path:
        assert args.prompt, "Prompt required for single video generation"

    args.data_resolution = [640, 1024]
    assert int(os.getenv("WORLD_SIZE", 1)) <= 1, "Single GPU inference only"
    
    Path(args.video_save_folder).mkdir(parents=True, exist_ok=True)
    
    return SamplingConfig(
        echo=True,
        temperature=args.temperature,
        top_p=args.top_p,
        compile_sampling=True,
    )

def load_vision_input(input_type: str, batch_input_path: str,
                     input_image_or_video_path: str, data_resolution: List[int],
                     num_input_frames: int, use_first_frames: bool = False):
    """Load visual inputs based on type."""
    input_type = input_type.replace("text_and_", "")
    loader_map = {
        "image": (read_input_images, read_input_image),
        "video": (lambda bp, dr, nf, uff: read_input_videos(bp, dr, num_input_frames, uff),
                 lambda p, dr, nf, uff: read_input_video(p, dr, num_input_frames, uff)),
        "array": (lambda bp, dr, nf, uff: read_input_arrays(bp, dr, num_input_frames, uff),
                 lambda p, dr, nf, uff: read_input_array(p, dr, num_input_frames, uff))
    }

    if batch_input_path:
        log.info(f"Loading batch inputs from {batch_input_path}")
        loader = loader_map[input_type][0]
        return loader(batch_input_path, data_resolution, num_input_frames, use_first_frames)
    else:
        loader = loader_map[input_type][1]
        return loader(input_image_or_video_path, data_resolution, num_input_frames, use_first_frames)

def prepare_video_batch_for_saving(video_batch: List[torch.Tensor]) -> List[np.ndarray]:
    """Convert output tensors to numpy arrays for saving."""
    return [(video * (255 if NORMALIZE_TENSOR else 1)).to(torch.uint8).permute(1, 2, 3, 0).cpu().numpy()
           for video in video_batch]

def prepare_array_batch(array_batch: List[torch.Tensor]) -> List[np.ndarray]:
    """Convert output tensors to numpy arrays for saving."""
    return [array.to(torch.float32).cpu().numpy() for array in array_batch]
