from argparse import ArgumentParser
import torch
import numpy as np


from cosmos1.models.autoregressive.tokenizer.discrete_video import DiscreteVideoFSQJITTokenizer
from cosmos1.models.autoregressive.tokenizer.training.data_loader import RandomHDF5Dataset, create_dataloader
from well_utils.data_processing.visualizations import create_video_heatmap


TOKENIZER_COMPRESSION_FACTOR = [8, 16, 16]
NUM_CONTEXT_FRAMES = 33
NUM_INPUT_FRAMES_VIDEO = 9
LATENT_SHAPE = [5, 40, 64]
DATA_RESOLUTION = [640, 1024]

CHANNEL_NAMES = ['tracer', 'velocity_x', 'velocity_y']

def main(args):
    T, H, W = LATENT_SHAPE

    data_loader = create_dataloader(
        data_dir=args.data_path,
        n_frames=NUM_CONTEXT_FRAMES,
        batch_size=1,
        num_workers=1,
    )
    next(iter(data_loader))
    sample = next(iter(data_loader)).cuda().permute(0, 4, 1, 2, 3)

    create_video_heatmap(sample[0].cpu().numpy(), save_path="original_5.mp4", fps=25, channel_names=CHANNEL_NAMES)

    video_tokenizer = DiscreteVideoFSQJITTokenizer(
        enc_fp=args.encoder_path,
        dec_fp=args.decoder_path,
        name="discrete_video_fsq",
        pixel_chunk_duration=NUM_CONTEXT_FRAMES,
        latent_chunk_duration=T,
    ).cuda()

    quantized_out, indices = video_tokenizer.encode(sample, pixel_chunk_duration=None)

    print("Quantized out shape:", quantized_out.shape, "Indices shape:", indices.shape)

    reconstructed = video_tokenizer.decode(indices)

    print("Reconstructed shape:", reconstructed.shape)

    print("Channel differences:", (sample - reconstructed).sum(dim=(0, 2, 3, 4)), "Mean channel differences:", (sample - reconstructed).mean(dim=(0, 2, 3, 4)))

    print("Absolute difference:", torch.abs(sample - reconstructed).sum(), "Mean absolute difference:", torch.abs(sample - reconstructed).mean())

    print("Mean percentage difference:", (torch.abs(sample - reconstructed) / sample).mean())

    print("MSE:", torch.nn.functional.mse_loss(sample, reconstructed))

    numpy_reconstructed = reconstructed[0].cpu().to(torch.float).numpy()

    print("Reconstructed shape:", numpy_reconstructed.shape, "Datatype:", numpy_reconstructed.dtype)

    create_video_heatmap(numpy_reconstructed, save_path="reconstructed_5.mp4", fps=25, channel_names=CHANNEL_NAMES)

    create_video_heatmap(torch.cat((sample, reconstructed), dim=0).cpu().numpy(), save_path="original_reconstructed5.mp4", fps=25, channel_names=CHANNEL_NAMES)

    numpy_video = create_video_heatmap(torch.cat((sample, reconstructed), dim=0).cpu().numpy(), fps=25, channel_names=CHANNEL_NAMES)
    from well_utils.data_processing.helpers import save_numpy_as_mp4

    print("SHape:", numpy_video.shape)

    save_numpy_as_mp4(numpy_video, "original_reconstructed6.mp4", fps=25)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_path", required=True, type=str, help="The path to the .bin .idx files")
    parser.add_argument("--encoder_path", required=True, type=str, help="The path to the encoder")
    parser.add_argument("--decoder_path", required=True, type=str, help="The path to the decoder")

    args = parser.parse_args()

    main(args)