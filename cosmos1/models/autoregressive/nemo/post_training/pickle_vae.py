import torch
import argparse
import yaml
from cosmo_lightning.models.vae_module import VAEModule
from the_well.data_processing.normalization.torch_normalize import TorchNormalizationApplier
from the_well.metrics.spatial import MSE, VRMSE
from the_well.data_processing.visualizations import create_video_heatmap

def _init_vae(args):
    config = yaml.load(open(args.config_path, "r"), Loader=yaml.SafeLoader)
    config['model']['pretrained_path'] = args.vae_checkpoint_path
    vae = VAEModule(**config['model'])
    vae.to(torch.bfloat16).to("cuda:0").eval()
    return vae.model

def _pickle_vae(vae, path):
    torch.save(vae, path)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--vae_checkpoint_path", type=str, required=True)
    args.add_argument("--config_path", type=str, required=True)
    args.add_argument("--output_path", type=str, required=True)
    args = args.parse_args()
    vae = _init_vae(args)
    _pickle_vae(vae, args.output_path)


"""
python -m cosmos1.models.autoregressive.nemo.post_training.pickle_vae \
    --vae_checkpoint_path /data0/arshkon/checkpoints/cosmos/finetuned2/tokenizers/turbulent_radiative_layer_2D_discrete/TRL2_cont2/checkpoints/epoch_221.ckpt \
    --config_path lightning_configs/pretrained_discreteTRL2.yaml \
    --output_path /data0/arshkon/checkpoints/cosmos/finetuned2/tokenizers/turbulent_radiative_layer_2D_discrete/TRL2_cont2/checkpoints/epoch_221.pth
"""