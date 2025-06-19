import numpy as np
import torch
import wandb
from pathlib import Path
from typing import Union, Dict, List
import os
import random
from lightning import LightningModule
from cosmo_lightning.utils.lr_scheduler import LinearWarmupCosineAnnealingLR
from cosmos1.models.autoregressive.tokenizer.networks import NetworkEval
from the_well.data_processing.visualizations import create_video_heatmap
from the_well.data_processing.helpers import resize_video_array
from the_well.metrics.spatial import VRMSE
from cosmo_lightning.utils.losses import build_vae_loss
from cosmo_lightning.models.tokenizer_factory import build_tokenizer


class VAEModule(LightningModule):
    def __init__(
        self,
        # vae parameters
        in_channels: int,
        out_channels: int,
        channels: int,
        channels_mult: List[int],
        z_channels: int,
        z_factor: int,
        embedding_dim: int,
        levels: List[int],
        spatial_compression: int,
        temporal_compression: int,
        num_res_blocks: int,
        patch_size: int,
        patch_method: str,
        num_groups: int,
        resolution: int,
        attn_resolutions: List[int],
        dropout: float,
        legacy_mode: bool = False,
        pretrained_path=None,
        scratch: bool = False,
        data_path: str = None,
        visual_log_interval=1,
        # continuous parameters
        mode: str = "discrete",
        latent_channels: int = 16,
        formulation: str = "AE",
        encoder: str = "Default",
        decoder: str = "Default",
        kl_beta: float = 1.0,
        # optimization parameters
        loss_type: str = 'l1',
        lr: float = 5e-4,
        beta_1: float = 0.9,
        beta_2: float = 0.99,
        weight_decay: float = 1e-5,
        warmup_epochs: int = 10,
        max_epochs: int = 100,
        warmup_start_lr: float = 1e-8,
        eta_min: float = 1e-8,
        constant_channels: List[int] = [],
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.model = build_tokenizer(self.hparams)
        
        if pretrained_path is not None:
            self._load_pretrained_weights(pretrained_path)
        
        self.criterion = build_vae_loss(
            mode=self.hparams.mode,
            beta=self.hparams.kl_beta,
            recon_loss_type=self.hparams.loss_type,
            normalization_path=os.path.join(self.hparams.data_path, 'normalization_stats.json'),
            formulation=self.hparams.formulation,
            constant_channels=self.hparams.constant_channels,
        )
    
    def _load_pretrained_dict(self, pretrained_path):
        print(f"Loading pretrained weights from {pretrained_path}")
        model_state_dict = torch.load(pretrained_path, weights_only=True, map_location=self.device)
        model_state_dict = {k.replace('model.', ''): v for k, v in model_state_dict.items()}
        self.model.load_state_dict(model_state_dict, strict=False)

    def _load_pretrained_weights(self, pretrained_path):   
        print(f"Loading pretrained model from {pretrained_path}")
        model = torch.load(pretrained_path, weights_only=False, map_location=self.device)
        if isinstance(model, dict):
            if 'state_dict' in model: 
                model_state_dict = model['state_dict']
            else: 
                model_state_dict = model
        else: 
            model_state_dict = model.state_dict()
        model_state_dict = {k.replace('model.', ''): v for k, v in model_state_dict.items()}
        missing_keys, unexpected_keys = self.model.load_state_dict(model_state_dict, strict=False)
        print(f"Missing VAE keys: {missing_keys}")
        print(f"Unexpected VAE keys: {unexpected_keys}")

    def _compute_grad_norm(self):
        total_norm_sq = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                total_norm_sq += p.grad.data.norm(2).item() ** 2
        return total_norm_sq ** 0.5
    
    def training_step(self, batch, batch_idx):
        output = self.model(batch)
        loss = self.criterion(output, batch)
        
        metric_dict = self.criterion.get_metrics(output, batch, split='train', prefix='')
        grad_norm = self._compute_grad_norm()
        metric_dict['train/grad_norm'] = grad_norm
        
        self.log_dict(
            metric_dict,
            prog_bar=True,
            on_step=True,
            on_epoch=False,
            batch_size=batch.size(0),
        )
        
        return loss
    
    def on_validation_epoch_start(self):
        self.sample_pair = (None, None)
    
    def validation_step(self, batch, batch_idx):
        output = self.model(batch)._asdict()
        norm_metric_dict = self.criterion.get_metrics(output, batch, split='val', prefix='norm_', denormalize=False)
        raw_metric_dict = self.criterion.get_metrics(output, batch, split='val', prefix='raw_', denormalize=True)
        metric_dict = {**norm_metric_dict, **raw_metric_dict}
        
        self.log_dict(
            metric_dict,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=batch.size(0),
            sync_dist=True,
        )
        
        # if first batch and gpu 0, save a sample pair
        if batch_idx == 0 and self.global_rank == 0:
            self.sample_pair = (random.choice(batch), output['reconstructions'][0])
    
    def on_validation_epoch_end(self):
        if self.global_rank == 0 and self.trainer.current_epoch % self.hparams.visual_log_interval == 0:
            original, reconstructed = self.sample_pair
            if self.hparams.constant_channels is not None:
                non_constant_channels = [i for i in range(self.hparams.in_channels) if i not in self.hparams.constant_channels]
                original = original[..., non_constant_channels, :, :, :]
            combined = torch.stack([original, reconstructed], dim=0)
            video_np = create_video_heatmap(combined.cpu().to(torch.float32).detach().numpy(), channel_names=[self.trainer.datamodule.hparams.channel_names[i] for i in non_constant_channels])
            resized_video_np = resize_video_array(video_np, width=512)
            resized_video_np = np.clip(np.expand_dims(resized_video_np.transpose(0, 3, 1, 2), axis=0).astype(np.uint8), 0, 255)
            # save video to disk
            # video_path = os.path.join(f"media/comparison_{self.trainer.current_epoch%20}.mp4")
            # create_video_heatmap(combined.cpu().to(torch.float32).detach().numpy(), channel_names=[self.trainer.datamodule.hparams.channel_names[i] for i in non_constant_channels], save_path=video_path, fps=25)
            self.logger.experiment.log({f"media/comparison": wandb.Video(resized_video_np, fps=25, format="mp4")})
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.hparams.lr,
            betas=(self.hparams.beta_1, self.hparams.beta_2),
            weight_decay=1e-5
        )

        n_steps_per_machine = len(self.trainer.datamodule.train_dataloader())
        n_steps = int(n_steps_per_machine / (self.trainer.num_devices * self.trainer.num_nodes))
        lr_scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            self.hparams.warmup_epochs * n_steps,
            self.hparams.max_epochs * n_steps,
            self.hparams.warmup_start_lr,
            self.hparams.eta_min,
        )
        scheduler = {"scheduler": lr_scheduler, "interval": "step", "frequency": 1}
        return {"optimizer": optimizer, "lr_scheduler": scheduler}