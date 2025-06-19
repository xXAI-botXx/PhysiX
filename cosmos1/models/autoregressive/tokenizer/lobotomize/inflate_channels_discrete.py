from pathlib import Path
from argparse import ArgumentParser
from typing import Optional

import torch
from omegaconf import OmegaConf
from hydra.utils import instantiate
from torch import nn

from cosmos1.models.autoregressive.tokenizer.lobotomize.helpers import inflate_channel_weights
from cosmos1.models.autoregressive.tokenizer.discrete_video import DiscreteVideoFSQJITTokenizer, DiscreteVideoFSQStateDictTokenizer
from cosmos1.models.autoregressive.configs.base.tokenizer import create_discrete_video_fsq_tokenizer_state_dict_config

device = torch.device("cuda")

def load_autoencoder_with_config(
    ckpt_dir: Path,
) -> nn.Module:
    """Load autoencoder by reconstructing architecture from config and loading JIT weights."""
    tokenizer_config = create_discrete_video_fsq_tokenizer_state_dict_config(
        ckpt_path=str(ckpt_dir / "ema.jit"),
        pixel_chunk_duration=33,
        compression_ratio=[8, 16, 16]
    )
    # tokenizer_config.tokenizer_module.patch_size = 2
    # tokenizer_config.tokenizer_module.temporal_compression = 1
    # tokenizer_config.tokenizer_module.compression_ratio = [1, 16, 16]
    # tokenizer_config.tokenizer_module.spatial_compression = 1

    autoencoder = instantiate(tokenizer_config.tokenizer_module)
    autoencoder.to(device=device, dtype=torch.bfloat16)

    # Determine JIT file paths
    suffix = ".jit"
    encoder_path = ckpt_dir / f"encoder{suffix}"
    decoder_path = ckpt_dir / f"decoder{suffix}"

    # Load state dicts
    encoder_state_dict = torch.jit.load(encoder_path).state_dict()
    decoder_state_dict = torch.jit.load(decoder_path).state_dict()
    combined_state_dict = {**encoder_state_dict, **decoder_state_dict}

    # Validate state dict
    autoencoder_state_dict = autoencoder.state_dict()
    missing = [k for k in autoencoder_state_dict if k not in combined_state_dict]
    extra = [k for k in combined_state_dict if k not in autoencoder_state_dict]

    if missing:
        print(f"Missing keys: {missing}")
    if extra:
        print(f"Extra keys: {extra}")

    autoencoder.load_state_dict(combined_state_dict, strict=False)
    return autoencoder

def save_model_components(
    autoencoder: nn.Module,
    save_dir: Path,
    input_channel_count: int,
    output_channel_count: int,
) -> Path:
    """Save full autoencoder model and return path to saved file."""
    save_path = save_dir / f"autoencoder_in_{input_channel_count}c_out_{output_channel_count}c.pt"
    torch.save(autoencoder, save_path)
    return save_path

def modify_encoder_channels(
    encoder: nn.Module,
    original_channels: int,
    new_channels: int,
) -> nn.Module:
    for name, param in encoder.named_parameters():
        if name == 'conv_in.0.conv3d.weight':
            if original_channels != new_channels:
                param.data = inflate_channel_weights(
                    param.data, original_channels, new_channels,
                    modify_input_shape=True, 
                )
    return encoder

def modify_decoder_channels(
    decoder: nn.Module,
    original_channels: int,
    new_channels: int,
) -> nn.Module:
    for name, param in decoder.named_parameters():
        if name in {'conv_out.1.conv3d.weight', 'conv_out.1.conv3d.bias'}:
            if original_channels != new_channels:
                param.data = inflate_channel_weights(
                    param.data, original_channels, new_channels,
                    modify_output_shape=True,
                )
    return decoder

def validate_model_shapes(
    autoencoder: nn.Module,
    new_input_channels: int,
    new_output_channels: int,
    dimensions: tuple,
):
    mock_input = torch.randn(1, new_input_channels, *dimensions, dtype=torch.bfloat16, device=device)
    mock_output = torch.randn(1, new_output_channels, *dimensions, dtype=torch.bfloat16, device=device)
    outputs = autoencoder(mock_input)
    assert outputs['reconstructions'].shape == mock_output.shape, \
        f"Shape mismatch: {outputs['reconstructions'].shape} vs {mock_input.shape}"

def test_saved_model(
    model_path: Path,
    new_input_channels: int,
    new_output_channels: int,
    dimensions: tuple,
):
    model = torch.load(model_path, weights_only=False).to(device=device)
    
    mock_input = torch.randn(1, new_input_channels, *dimensions, dtype=torch.bfloat16, device=device)
    mock_output = torch.randn(1, new_output_channels, *dimensions, dtype=torch.bfloat16, device=device)
    outputs = model(mock_input)
    
    assert outputs['reconstructions'].shape == mock_output.shape, \
        f"Shape mismatch: {outputs['reconstructions'].shape} vs {mock_input.shape}"
    print("Max value: ", outputs['reconstructions'].max().item(), "Min value: ", outputs['reconstructions'].min().item())
    print("Mean percent difference: ", torch.mean(torch.abs(outputs['reconstructions'] - mock_output) / mock_output).item())
    print("Test passed successfully!")

def inflate_channels(
    autoencoder_dir: Path,
    original_channels: int,
    new_input_channels: int,
    new_output_channels: int,
    dimensions: tuple,
):
    # Load base model
    autoencoder = load_autoencoder_with_config(autoencoder_dir)

    # Modify channel dimensions
    autoencoder.encoder = modify_encoder_channels(autoencoder.encoder, original_channels, new_input_channels)
    autoencoder.decoder = modify_decoder_channels(autoencoder.decoder, original_channels, new_output_channels)

    # Validate before saving
    validate_model_shapes(autoencoder, new_input_channels, new_output_channels, dimensions)
    # Save and test
    saved_model_path = save_model_components(autoencoder, autoencoder_dir, new_input_channels, new_output_channels)
    test_saved_model(saved_model_path, new_input_channels, new_output_channels, dimensions)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--autoencoder_path", type=Path, required=True)
    parser.add_argument("--original_channels", type=int, required=True)
    parser.add_argument("--new_input_channels", type=int, required=True)
    parser.add_argument("--new_output_channels", type=int, required=True)
    parser.add_argument("--dimensions", type=int, nargs=3, required=True)
    args = parser.parse_args()

    inflate_channels(
        args.autoencoder_path,
        args.original_channels,
        args.new_input_channels,
        args.new_output_channels,
        tuple(args.dimensions)
    )

"""
python -m cosmos1.models.autoregressive.tokenizer.lobotomize.inflate_channels_discrete \
    --autoencoder_path /data0/arshkon/checkpoints/cosmos/Cosmos-1.0-Tokenizer-DV8x16x16 \
    --original_channels 3 \
    --new_input_channels 5 \
    --new_output_channels 3 \
    --dimensions 33 256 256
"""
