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
import omegaconf

import torch
from einops import rearrange
from nemo.collections.nlp.data.language_modeling.megatron import indexed_dataset

from cosmos1.models.autoregressive.nemo.utils import read_input_videos, read_input_array
from cosmos1.utils import log
from cosmo_lightning.models.universal_multi_decoder_vae_module import UniversalMultiDecoderVAEModule


data_readers = {
    "video": read_input_videos,
    "video_array": read_input_array
}


def main(args):
    config_path = os.path.join(args.autoencoder_path, "config.yaml")
    # read the config file
    config = omegaconf.OmegaConf.load(config_path)
    
    lightning_module = UniversalMultiDecoderVAEModule(**config['model'])
    
    ckpt_paths = glob(os.path.join(args.autoencoder_path, "checkpoints", "epoch_*.ckpt"))
    assert len(ckpt_paths) == 1, f"There should be only one best checkpoint in {args.autoencoder_path}/checkpoints"
    ckpt = torch.load(ckpt_paths[0], map_location="cpu")
    state_dict = ckpt['state_dict']
    msg = lightning_module.load_state_dict(state_dict, strict=True)
    print(f"Loaded state dict from {ckpt_paths[0]}: {msg}")
    
    video_tokenizer = lightning_module.model
    video_tokenizer.to("cuda").eval()
    
    # Enable bfloat16 autocast globally
    torch.set_float32_matmul_precision('high')
    for split in ["train", "valid", "test"]:
        input_videos_dir = os.path.join(args.input_videos_dir, split)
        output_prefix = os.path.join(args.output_prefix, split, 'embeddings')
        os.makedirs(os.path.dirname(output_prefix), exist_ok=True)
        
        builders = {}
        key = "text"
        builders[key] = indexed_dataset.make_builder(
            f"{output_prefix}.bin",
            impl="mmap",
            chunk_size=64,
            pad_id=0,
            retrieval_db=None,
            vocab_size=64000,
            stride=64,
        )

        filepaths_final = glob(f"{input_videos_dir}/*.{args.data_suffix}")
        iterator = tqdm(filepaths_final, desc="Processing videos")
        dataset_name = None
        for name in config['data']['metadata_dict'].keys():
            if name in input_videos_dir:
                dataset_name = name
                break
        channel_names = config['data']['metadata_dict'][dataset_name]['channel_names']

        debug = args.debug

        for filepath in iterator:
            # input_video shape: [total_videos, C, T, H, W]
            input_video = data_readers[args.data_reader](filepath, args.dimensions, sliding_windows=True)
            total_videos = input_video.shape[0]
            
            # Process in batches
            for batch_idx in range(0, total_videos, args.batch_size):
                batch_end = min(batch_idx + args.batch_size, total_videos)
                video_batch = input_video[batch_idx:batch_end]
                video_batch = video_batch.cuda()
                
                with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    (quant_info, quant_codes, _), _ = video_tokenizer.encode(video_batch, dataset_name, channel_names)
                    if debug:
                        reconstructed_video = video_tokenizer.decode(quant_codes, dataset_name)

                if debug:
                    from well_utils.data_processing.visualizations import create_video_heatmap
                    print("Quant info shape:", quant_info.shape)
                    create_video_heatmap(reconstructed_video[:3].detach().cpu().to(torch.float32).numpy(), save_path=f"reconstructed_video_debug.mp4", fps=25)
                    debug = False
                
                # Add each video in the batch individually
                for video_idx in range(quant_info.shape[0]):
                    single_video_quant = quant_info[video_idx]  # [T, H, W]
                    indices = rearrange(single_video_quant, "T H W -> (T H W)").detach().cpu()
                    builders[key].add_item(torch.IntTensor(indices))
                    builders[key].end_document()

        builders[key].finalize(f"{output_prefix}.idx")
        log.info(f"Stored the .bin and .idx files in {output_prefix}")



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
        default="hdf5",
        type=str,
        help="The suffix of the input video files (e.g .mp4)",
    )
    parser.add_argument(
        "--data_reader",
        required=False,
        default="video_array",
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
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode to visualize the reconstructed video",
    )
    args = parser.parse_args()

    main(args)