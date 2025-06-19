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

import os
from argparse import ArgumentParser
from glob import glob
from tqdm import tqdm

import torch
from einops import rearrange
from huggingface_hub import snapshot_download
from nemo.collections.nlp.data.language_modeling.megatron import indexed_dataset

from cosmos1.models.autoregressive.nemo.utils import read_input_videos, read_input_array
from cosmos1.models.autoregressive.tokenizer.discrete_video import DiscreteVideoFSQJITTokenizer
from cosmos1.utils import log

TOKENIZER_COMPRESSION_FACTOR = [8, 16, 16]
NUM_CONTEXT_FRAMES = 33

data_readers = {"video": read_input_videos,
                "video_array": read_input_array}


def main(args):
    video_tokenizer = torch.load(args.autoencoder_path, weights_only=False)
    video_tokenizer.to("cuda").to(torch.bfloat16).eval()

    builders = {}
    key = "text"
    builders[key] = indexed_dataset.make_builder(
        f"{args.output_prefix}.bin",
        impl="mmap",
        chunk_size=64,
        pad_id=0,
        retrieval_db=None,
        vocab_size=64000,
        stride=64,
    )

    filepaths_final = glob(f"{args.input_videos_dir}/*.{args.data_suffix}")
    iterator = tqdm(filepaths_final, desc="Processing videos")

    debug = True

    for filepath in iterator:
        # input_video shape: [total_videos, C, T, H, W]
        input_video = data_readers[args.data_reader](filepath, args.dimensions, sliding_windows=True).to(torch.bfloat16)
        total_videos = input_video.shape[0]
        
        # Process in batches
        for batch_idx in range(0, total_videos, args.batch_size):
            batch_end = min(batch_idx + args.batch_size, total_videos)
            video_batch = input_video[batch_idx:batch_end]
            video_batch = video_batch.cuda().to(torch.bfloat16)
            
            with torch.no_grad():
                quant_info, quant_codes, quant_loss = video_tokenizer.encode(video_batch)

            if debug:
                from the_well.data_processing.visualizations import create_video_heatmap
                print("Quant info shape:", quant_info.shape)
                reconstructed_video = video_tokenizer.decode(quant_codes)
                create_video_heatmap(reconstructed_video[:3].detach().cpu().to(torch.float32).numpy(), save_path=f"reconstructed_video_debug.mp4", fps=25)
                debug = False
            
            # Add each video in the batch individually
            for video_idx in range(quant_info.shape[0]):
                single_video_quant = quant_info[video_idx]  # [T, H, W]
                indices = rearrange(single_video_quant, "T H W -> (T H W)").detach().cpu()
                builders[key].add_item(torch.IntTensor(indices))
                builders[key].end_document()

    builders[key].finalize(f"{args.output_prefix}.idx")
    log.info(f"Stored the .bin and .idx files in {args.output_prefix}")



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input_videos_dir", required=True, type=str, help="The path to the input videos")
    parser.add_argument(
        "--autoencoder_path",
        required=True,
        type=str,
        help="The path to the autoencoder",
    )
    parser.add_argument(
        "--output_prefix",
        required=True,
        type=str,
        help="The directory along with the output file name to write the .idx and .bin files (e.g /path/to/output/sample)",
    )
    parser.add_argument(
        "--data_suffix",
        required=False,
        default="mp4",
        type=str,
        help="The suffix of the input video files (e.g .mp4)",
    )
    parser.add_argument(
        "--data_reader",
        required=False,
        default="video",
        type=str,
        help="The data reader to use",
    )
    parser.add_argument(
        "--dimensions",
        required=False,
        nargs=2,
        default=[256, 512],
        type=int,
        help="The dimensions to resize the input video to",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for processing videos through the model",
    )
    args = parser.parse_args()

    with torch.no_grad():
        main(args)