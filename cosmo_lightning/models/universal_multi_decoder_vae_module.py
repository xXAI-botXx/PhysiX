import os
import torch
from typing import List, Union
from lightning import LightningModule
from cosmo_lightning.utils.lr_scheduler import LinearWarmupCosineAnnealingLR
from well_utils.data_processing.visualizations import create_video_heatmap
from well_utils.data_processing.helpers import resize_video_array
from well_utils.metrics.spatial import VRMSE
from cosmo_lightning.utils.losses import build_vae_loss
from cosmos1.models.autoregressive.tokenizer.universal_multi_decoder_tokenizer import UniversalMultiDecoderCausalDiscreteVideoTokenizer

IGNORE_CHANNELS = [
    'mask_HS',
    'density_ASD', 'density_ASI', 'density_ASM',
    'speed_of_sound_ASD', 'speed_of_sound_ASI', 'speed_of_sound_ASM',
]
IGNORE_DATASETS = [
    'euler_multi_quadrants_periodicBC',
]

class UniversalMultiDecoderVAEModule(LightningModule):
    def __init__(
        self,
        data_metadata: dict,
        learnable_padding: Union[bool, None],
        max_video_size: Union[List[int], None],
        film: Union[bool, None],
        film_embedding: Union[int, None],
        z_channels: int,
        z_factor: int,
        patch_size: int,
        patch_method: str,
        channels: int,
        channels_mult: List[int],
        embedding_dim: int,
        levels: List[int],
        spatial_compression: int,
        temporal_compression: int,
        num_res_blocks: int,
        resolution: int,
        attn_resolutions: List[int],
        dropout: float = 0.0,
        legacy_mode: bool = False,
        # pretrained weights and logging parameters
        pretrained_path=None,
        stage_one_path=None,
        visual_log_interval=1,
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
        use_gradient_checkpoint:bool = False,
        accumulate_grad_batches:int = 1,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.accumulate_grad_batches = accumulate_grad_batches
        
        self.model = UniversalMultiDecoderCausalDiscreteVideoTokenizer(
            data_metadata=data_metadata,
            learnable_padding=learnable_padding,
            max_video_size=max_video_size,
            film=film,
            film_embedding=film_embedding,
            z_channels=z_channels,
            z_factor=z_factor,
            patch_size=patch_size,
            patch_method=patch_method,
            channels=channels,
            channels_mult=channels_mult,
            embedding_dim=embedding_dim,
            levels=levels,
            spatial_compression=spatial_compression,
            temporal_compression=temporal_compression,
            num_res_blocks=num_res_blocks,
            resolution=resolution,
            attn_resolutions=attn_resolutions,
            dropout=dropout,
            legacy_mode=legacy_mode,
        )
        
        if stage_one_path is not None:
            state_dict = torch.load(stage_one_path, map_location=self.device)['state_dict']
            msg = self.load_state_dict(state_dict, strict=True)
            print("Loading stage one weights", msg)
            pretrained_path = None
        
        if pretrained_path is not None:
            self._load_pretrained_weights(pretrained_path, data_metadata)
        
        self.criterion = build_vae_loss(
            mode='discrete',
            recon_loss_type=loss_type,
        )

    def _load_pretrained_weights(self, pretrained_path, data_metadata):
        state_dict = torch.load(pretrained_path, map_location=self.device).state_dict()
        
        quant_conv_state_dict = {k: v for k, v in state_dict.items() if 'quant_conv.' in k and 'post_quant_conv.' not in k}
        quant_conv_state_dict = {k.replace('quant_conv.', ''): v for k, v in quant_conv_state_dict.items()}
        msg = self.model.quant_conv.load_state_dict(quant_conv_state_dict, strict=True)
        print("Loading quant_conv", msg)
        
        post_quant_conv_state_dict = {k: v for k, v in state_dict.items() if 'post_quant_conv.' in k}
        post_quant_conv_state_dict = {k.replace('post_quant_conv.', ''): v for k, v in post_quant_conv_state_dict.items()}
        msg = self.model.post_quant_conv.load_state_dict(post_quant_conv_state_dict, strict=True)
        print("Loading post_quant_conv", msg)
            
        encoder_state_dict = {k: v for k, v in state_dict.items() if 'encoder.' in k}
        if (encoder_state_dict['encoder.conv_in.0.conv3d.weight'].shape[0] != self.model.encoder.conv_in[0].conv3d.weight.shape[0]) or \
           (encoder_state_dict['encoder.conv_in.0.conv3d.weight'].shape[1] != self.model.encoder.conv_in[0].conv3d.weight.shape[1]):
            encoder_state_dict = {k: v for k, v in state_dict.items() if 'encoder.conv_in.0.' not in k}
        encoder_state_dict = {k.replace('encoder.', ''): v for k, v in encoder_state_dict.items()}
        msg = self.model.encoder.load_state_dict(encoder_state_dict, strict=False)
        print("Loading encoder", msg)
        
        for dataset in data_metadata.keys():
            # load data-specific decoder
            decoder_state_dict = {k: v for k, v in state_dict.items() if 'decoder.' in k}
            if (decoder_state_dict['decoder.conv_out.0.conv3d.weight'].shape[0] != self.model.decoder[dataset].conv_out[0].conv3d.weight.shape[0]) or \
               (decoder_state_dict['decoder.conv_out.0.conv3d.weight'].shape[1] != self.model.decoder[dataset].conv_out[0].conv3d.weight.shape[1]):
                decoder_state_dict = {k: v for k, v in state_dict.items() if 'decoder.conv_out.' not in k}
            decoder_state_dict = {k.replace('decoder.', ''): v for k, v in decoder_state_dict.items()}
            msg = self.model.decoder[dataset].load_state_dict(decoder_state_dict, strict=False)
            print(f"Loading decoder for {dataset}", msg)

    def _compute_grad_norm(self):
        total_norm_sq = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                total_norm_sq += p.grad.data.norm(2).item() ** 2
        return total_norm_sq ** 0.5
    
    def training_step(self, batches, batch_idx):
        loss = 0.
        opt = self.optimizers()
        # opt.zero_grad()
        # random selection
        sch = self.lr_schedulers()
        for batch in batches:
            videos, dataset_name, channel_names,padding_length = batch
            if dataset_name in IGNORE_DATASETS:
                continue
            output = self.model(videos, dataset_name, channel_names, checkpointing=self.hparams.use_gradient_checkpoint)
            
            # ignore constant channels
            assert len(channel_names) == videos.shape[1] == output['reconstructions'].shape[1]
            valid_indices = [i for i, name in enumerate(channel_names) if name not in IGNORE_CHANNELS]
            output['reconstructions'] = output['reconstructions'][:, valid_indices, :, :, :]
            videos = videos[:, valid_indices, :, :, :]
            
            dataset_loss = self.criterion(output, videos)
            self.manual_backward(dataset_loss)
            loss += dataset_loss.detach()
            metric_dict = self.criterion.get_metrics(output, videos, split='train', prefix=f'{dataset_name}_')
        
            grad_norm = self._compute_grad_norm()
            metric_dict[f'train/{dataset_name}_grad_norm'] = grad_norm
            # breakpoint()
            metric_dict[f'schedule_epoch'] = sch._step_count / sch.max_epochs * self.hparams.max_epochs
            self.log_dict(
                metric_dict,
                prog_bar=True,
                on_step=True,
                on_epoch=False,
                batch_size=videos.size(0),
            )
        #print(batch_idx,self.accumulate_grad_batches,self._nsteps)
        if (batch_idx + 1) % self.accumulate_grad_batches == 0:
            self.clip_gradients(opt, gradient_clip_val=10, gradient_clip_algorithm="norm")
            opt.step()
            opt.zero_grad()
            sch = self.lr_schedulers()
            sch.step()
            #print("scheduler step")
        
        # loss /= len(batches)
        # return loss
    
    # def on_validation_epoch_start(self):
    #     self.sample_pair = (None, None)
    
    def on_validation_epoch_start(self):
        opt = self.optimizers()
        opt.zero_grad()
        self.agg_loss = []
        self.agg_raw_vrmse = []
        
    def on_train_epoch_start(self):
        opt = self.optimizers()
        opt.zero_grad()
    
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        videos, dataset_name, channel_names,padding_length = batch
        output = self.model(videos, dataset_name, channel_names)._asdict()
        
        # ignore constant channels
        valid_indices = [i for i, name in enumerate(channel_names) if name not in IGNORE_CHANNELS]
        output['reconstructions'] = output['reconstructions'][:, valid_indices, :, :, :]
        videos = videos[:, valid_indices, :, :, :]
        
        loss = self.criterion(output, videos)
        self.agg_loss.append(loss.item())
        
        # normalized metric dict
        norm_metric_dict = self.criterion.get_metrics(output, videos, split='val', prefix=f'{dataset_name}_norm_', denormalize=False)
        
        # raw metric dict
        output['reconstructions'] = self.trainer.datamodule.denormalize(output['reconstructions'], dataset_name, valid_indices=valid_indices)
        videos = self.trainer.datamodule.denormalize(videos, dataset_name, valid_indices=valid_indices)
        raw_metric_dict = self.criterion.get_metrics(output, videos, split='val', prefix=f'{dataset_name}_raw_', denormalize=False)
        metric_dict = {**norm_metric_dict, **raw_metric_dict}
        if dataset_name not in IGNORE_DATASETS:
            self.agg_raw_vrmse.append(metric_dict[f'val/{dataset_name}_raw_VRMSE'])
        
        self.log_dict(
            metric_dict,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=videos.size(0),
            sync_dist=True,
            add_dataloader_idx=False,
        )
    
    def on_validation_epoch_end(self):
        # aggregate self.agg_loss across all processes
        local_mean_loss = torch.tensor(sum(self.agg_loss) / len(self.agg_loss)).to(self.device)
        local_raw_vrmse = torch.tensor(sum(self.agg_raw_vrmse) / len(self.agg_raw_vrmse)).to(self.device)
        
        torch.distributed.barrier()
        
        gathered_means = self.all_gather(local_mean_loss)
        gathered_raw_vrmse = self.all_gather(local_raw_vrmse)
        # Calculate the global mean across all ranks
        global_mean_loss = gathered_means.mean().item()
        global_raw_vrmse = gathered_raw_vrmse.mean().item()
        metric_dict = {
            'val/loss': global_mean_loss,
            'val/raw_VRMSE': global_raw_vrmse,
        }
        self.log_dict(
            metric_dict,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.hparams.lr,
            betas=(self.hparams.beta_1, self.hparams.beta_2),
            weight_decay=1e-5
        )

        n_steps_per_machine = len(iter(self.trainer.datamodule.train_dataloader()))
        n_steps = int(n_steps_per_machine / (self.trainer.num_devices * self.trainer.num_nodes) / self.accumulate_grad_batches)
        # 
        self._nsteps = dict(
            n_steps=n_steps,
            n_steps_per_machine=n_steps_per_machine,
            num_devices=self.trainer.num_devices,
            num_nodes=self.trainer.num_nodes,
            accumulate_grad_batches=self.accumulate_grad_batches,
        )
        n_steps = int(n_steps_per_machine / (self.trainer.num_devices * self.trainer.num_nodes) / self.accumulate_grad_batches)
        lr_scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            self.hparams.warmup_epochs * n_steps,
            self.hparams.max_epochs * n_steps,
            self.hparams.warmup_start_lr,
            self.hparams.eta_min,
        )
        scheduler = {"scheduler": lr_scheduler, "interval": "step", "frequency": 1}
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
    

# vae_module = UniversalVAEModule(
#     variables=[
#         "tracer",
#         "buoyancy",
#         "pressure",
#         "concentration",
#         "velocity_x",
#         "velocity_y",
#         "D_xx",
#         "D_xy",
#         "D_yx",
#         "D_yy",
#         "E_xx",
#         "E_xy",
#         "E_yx",
#         "E_yy",
#     ],
#     max_video_size=[33, 512, 512],
#     # patcher_type="cross_attn",
#     patcher_type="padded",
#     patch_emb_dim=1024,
#     patch_emb_nheads=16,
#     z_channels=16,
#     z_factor=1,
#     patch_size=4,
#     patch_method="haar",
#     channels=128,
#     channels_mult=[2, 4, 4],
#     num_res_blocks=2,
#     attn_resolutions=[32],
#     dropout=0.0,
#     resolution=1024,
#     spatial_compression=16,
#     temporal_compression=8,
#     legacy_mode=False,
#     levels=[8, 8, 8, 5, 5, 5],
#     embedding_dim=6,
#     pretrained_path='/eagle/MDClimSim/tungnd/physics_sim/cosmos_ckpts/Cosmos-1.0-Tokenizer-DV8x16x16/autoencoder_14c.pt',
# )
