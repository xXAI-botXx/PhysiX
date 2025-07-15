#!/usr/bin/env python
"""
This script loads the CausalContinuousVideoTokenizer JIT model,
inflates its input and output channel weights from an original channel
count to a new desired channel count, validates the modified model,
and then saves the updated model.
"""

import argparse
from pathlib import Path
import torch
import random

# Import the continuous tokenizer model and its default configuration.
from cosmos1.models.tokenizer.networks.continuous_video import CausalContinuousVideoTokenizer
from cosmos1.models.tokenizer.networks.configs import continuous_video

# Import a helper function to inflate channel weights.
from cosmos1.models.autoregressive.tokenizer.lobotomize.helpers import inflate_channel_weights


def remove_ddp_prefix(state_dict: dict) -> dict:
    """
    Remove 'module.' prefix from state_dict keys saved from a DDP model.
    """
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key[len("module."):] if key.startswith("module.") else key
        new_state_dict[new_key] = value
    return new_state_dict


def modify_input_channels(model: torch.nn.Module, original_channels: int, new_channels: int) -> torch.nn.Module:
    """
    Iterate over model parameters and inflate weights for input convolution layers.
    This implementation expects the input convolution parameters to have 'conv_in' in their name.
    """
    for name, param in model.named_parameters():
        if name == 'encoder.conv_in.0.conv3d.weight':
            print(f"Modifying input channels for parameter: {name}")
            param.data = inflate_channel_weights(param.data, original_channels, new_channels, modify_input_shape=True)
    return model


def modify_output_channels(model: torch.nn.Module, original_channels: int, new_channels: int) -> torch.nn.Module:
    """
    Iterate over model parameters and inflate weights for output convolution layers.
    This implementation expects the output convolution parameters to have 'conv_out' in their name.
    """
    for name, param in model.named_parameters():
        if name in {'decoder.conv_out.1.conv3d.weight', 'decoder.conv_out.1.conv3d.bias'}:
            print(f"Modifying output channels for parameter: {name}")
            param.data = inflate_channel_weights(param.data, original_channels, new_channels, modify_output_shape=True)
    return model


def validate_model(model: torch.nn.Module, new_channels: int, frames: int, height: int, width: int, device: torch.device):
    """
    Create a dummy input tensor and run a forward pass.
    Assert that the reconstruction shape matches the dummy input shape.
    """
    dummy_input = torch.randn(1, new_channels, frames, height, width, device=device)
    with torch.no_grad():
        output = model(dummy_input)

    # The model's output might be a dict or a namedtuple with attribute 'reconstructions'.
    if hasattr(output, "reconstructions"):
        rec = output.reconstructions
    elif isinstance(output, dict) and "reconstructions" in output:
        rec = output["reconstructions"]
    else:
        raise ValueError("Output of the model does not contain 'reconstructions'.")
        
    assert rec.shape == dummy_input.shape, f"Reconstructed shape {rec.shape} does not match input {dummy_input.shape}"
    print("Validation successful: reconstruction shape", rec.shape)


def save_model(model: torch.nn.Module, save_dir: Path, new_channels: int) -> Path:
    """
    Save the modified model to the parent directory of the provided weights file.
    """
    save_path = save_dir / f"vae_{new_channels}c.pt"
    torch.save(model, save_path)
    print("Saved inflated model to", save_path)
    return save_path


def test_saved_model(model_path: Path, new_channels: int, frames: int, height: int, width: int, device: torch.device):
    """
    Load the saved model and run a dummy forward pass.
    """
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.to(device)
    
    dummy_input = torch.randn(1, new_channels, frames, height, width, device=device)
    with torch.no_grad():
        output = model(dummy_input)

    print("Input shape:", dummy_input.shape)
    print("Latent shape:", output['latent'].shape)
    print("type of output['posteriors']", type(output['posteriors']))
    for i, x in enumerate(output['posteriors']):
        print(i, x.item())
    print("Reconstruction shape:", output['reconstructions'].shape)
        
    assert output['reconstructions'].shape == dummy_input.shape, f"Output shape {output['reconstructions'].shape} does not match input {dummy_input.shape}"
    print("Test passed")


def main():
    parser = argparse.ArgumentParser(
        description="Inflate channels for the Continuous Tokenizer model weights."
    )
    parser.add_argument(
        "--weights",
        type=Path,
        required=True,
        help="Path to the continuous tokenizer JIT model weights."
    )
    parser.add_argument(
        "--original_channels",
        type=int,
        required=True,
        help="Original channel count in the model (e.g. 3)."
    )
    parser.add_argument(
        "--new_channels",
        type=int,
        required=True,
        help="Desired new channel count."
    )
    parser.add_argument(
        "--frames", type=int, default=16, help="Number of video frames (default: 16)."
    )
    parser.add_argument(
        "--height", type=int, default=64, help="Input frame height (default: 64)."
    )
    parser.add_argument(
        "--width", type=int, default=64, help="Input frame width (default: 64)."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run the model (default: cuda if available)."
    )
    args = parser.parse_args()
    
    device = torch.device(args.device)
    
    # Instantiate the continuous tokenizer model.
    model = CausalContinuousVideoTokenizer(**continuous_video)
    model = model.to(device)
    
    # Load the JIT model and extract its state dict.
    jit_model = torch.jit.load(args.weights, map_location=device)
    state_dict = jit_model.state_dict()
    state_dict = remove_ddp_prefix(state_dict)
    
    # Load weights into the model non-strictly.
    load_result = model.load_state_dict(state_dict, strict=False)
    if load_result.missing_keys:
        print("Missing keys:", load_result.missing_keys)
    if load_result.unexpected_keys:
        print("Unexpected keys:", load_result.unexpected_keys)
    
    # Inflate (modify) the input and output channel dimensions.
    model = modify_input_channels(model, args.original_channels, args.new_channels)
    model = modify_output_channels(model, args.original_channels, args.new_channels)
    
    # Validate the modified model.
    validate_model(model, args.new_channels, args.frames, args.height, args.width, device)
    
    # Save the modified model.
    save_dir = args.weights.parent
    save_path = save_model(model, save_dir, args.new_channels)
    
    # Test the saved model.
    test_saved_model(save_path, args.new_channels, args.frames, args.height, args.width, device)


if __name__ == "__main__":
    main()
