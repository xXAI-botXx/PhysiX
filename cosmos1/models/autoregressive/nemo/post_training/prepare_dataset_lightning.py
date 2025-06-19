import os
from argparse import ArgumentParser
from glob import glob
import omegaconf
import torch
torch.set_float32_matmul_precision('high')
from lightning import Trainer
from lightning.pytorch.callbacks import RichModelSummary, TQDMProgressBar

from cosmo_lightning.models.universal_multi_decoder_vae_module import UniversalMultiDecoderVAEModule
from cosmo_lightning.data.gen_embedding_datamodule import GenEmbeddingDataModule
from cosmo_lightning.models.gen_embedding_module import GenEmbeddingModule


def main(args):
    config_path = os.path.join(args.autoencoder_path, "config.yaml")
    config = omegaconf.OmegaConf.load(config_path)
    lightning_module = UniversalMultiDecoderVAEModule(**config['model'])
        
    ckpt_paths = glob(os.path.join(args.autoencoder_path, "checkpoints", "epoch_*.ckpt"))
    assert len(ckpt_paths) == 1, f"There should be only one best checkpoint in {args.autoencoder_path}/checkpoints"
    ckpt = torch.load(ckpt_paths[0], map_location="cpu")
    state_dict = ckpt['state_dict']
    msg = lightning_module.load_state_dict(state_dict, strict=True)
    print(f"Loaded state dict from {ckpt_paths[0]}: {msg}")

    dataset_name = None
    for name in config['data']['metadata_dict'].keys():
        if name in args.input_videos_dir:
            dataset_name = name
            break
    channel_names = config['data']['metadata_dict'][dataset_name]['channel_names']
    
    gen_datamodule = GenEmbeddingDataModule(
        root_dir=args.input_videos_dir,
        dimensions=args.dimensions,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    gen_module = GenEmbeddingModule(
        vae_module=lightning_module,
        output_dir=args.output_dir,
    )
    gen_module.set_data_metadata(dataset_name, channel_names)

    trainer = Trainer(
        accelerator="gpu",
        strategy="ddp",
        num_nodes=args.num_nodes,
        devices=args.devices,
        precision="bf16-mixed",
        min_epochs=1,
        max_epochs=1,
        enable_progress_bar=True,
        callbacks=[
            # RichModelSummary(max_depth=-1),
            TQDMProgressBar(),
        ],
        logger=None,
        num_sanity_val_steps=0,
    )
    gen_datamodule.setup()
    trainer.fit(
        model=gen_module,
        train_dataloaders=gen_datamodule.train_dataloader(),
    )
    trainer.validate(
        model=gen_module,
        dataloaders=gen_datamodule.val_dataloader(),
    )
    trainer.test(
        model=gen_module,
        dataloaders=gen_datamodule.test_dataloader(),
    )
    

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
        "--output_dir",
        required=True,
        type=str,
        help="The directory along with the output file name to write the .idx and .bin files (e.g /path/to/output/sample)",
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
        default=8,
        help="Batch size for processing videos through the model",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers for data loading",
    )
    parser.add_argument(
        "--num_nodes",
        type=int,
        default=1,
        help="Number of nodes for distributed training",
    )
    parser.add_argument(
        "--devices",
        type=int,
        default=1,
        help="Number of devices for distributed training",
    )
    args = parser.parse_args()

    main(args)