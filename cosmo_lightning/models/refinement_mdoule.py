import os
import torch
from typing import List, Union
from lightning import LightningModule
from cosmo_lightning.utils.lr_scheduler import LinearWarmupCosineAnnealingLR
from well_utils.data_processing.visualizations import create_video_heatmap
from well_utils.data_processing.helpers import resize_video_array
from well_utils.metrics.spatial import VRMSE
from cosmo_lightning.utils.losses import build_vae_loss
from cosmos1.models.autoregressive.tokenizer.universal_tokenizer import UniversalCausalDiscreteVideoTokenizer
# from the_well.benchmark
IGNORE_CHANNELS = ['mask_HS', 'density_AS', 'speed_of_sound_AS']
from typing import Dict
from cosmo_lightning.models.benchmark.convnext import UNetConvNext
# import defaultdict
import pandas as pd
import json
class RefinementModule(LightningModule):
    def __init__(
        self,
        model_config: Dict,
        context_frames: int = 5,
        grounding_frames: List[int] = [-1,0],
        pretrained_path=None,

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
        valid_indices = None,
        load_path = None,
        # normalization_path=None
      
        # automatic_optimization=False,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.accumulate_grad_batches = accumulate_grad_batches
        dim_out = model_config['dim_out']
        total_input_frames = context_frames + len(grounding_frames)
        dim_in =  model_config['dim_out'] * total_input_frames
        model_config['dim_in'] = dim_in
        print(f"Model Config: dim_in: {dim_in}, dim_out: {dim_out}")
        self.model = UNetConvNext(
            **model_config
        ) 
        if load_path:
            #
            # load safetey
            if 'safetensors' in load_path:
                from safetensors.torch import load_file
                ckpt = load_file(load_path,"cpu")
            else:
                ckpt = torch.load(load_path, map_location='cpu')
            # breakpoint()
            self.model.load_state_dict(ckpt, strict=True)
        if pretrained_path is not None:
            self._load_pretrained_weights(pretrained_path)
        
        self.criterion = build_vae_loss(
            mode='discrete',
            recon_loss_type=loss_type,
            # stats_path=normalization_path,
        )
        self.valid_indices = valid_indices

    def _load_pretrained_weights(self, pretrained_path):   
        print(f"Loading pretrained model from {pretrained_path}")
        if ".pt" in pretrained_path:
            model = torch.load(pretrained_path, weights_only=False, map_location=self.device)
            pretrained_state_dict = model.state_dict()
            msg = self.model.load_state_dict(pretrained_state_dict, strict=False)
            print(msg)
        else:
            encoder_state_dict = torch.jit.load(os.path.join(pretrained_path, 'encoder.jit')).state_dict()
            decoder_state_dict = torch.jit.load(os.path.join(pretrained_path, 'decoder.jit')).state_dict()
            pretrained_state_dict = {**encoder_state_dict, **decoder_state_dict}
            pretrained_keys = list(pretrained_state_dict.keys())
            for key in pretrained_keys:
                if key not in self.model.state_dict().keys() or self.model.state_dict()[key].shape != pretrained_state_dict[key].shape:
                    # print(f"Removing {key} in pretrained model")
                    # remove the key from the pretrained state dict
                    del pretrained_state_dict[key]
            msg = self.model.load_state_dict(pretrained_state_dict, strict=False)
            print(msg)

    def _compute_grad_norm(self):
        total_norm_sq = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                total_norm_sq += p.grad.data.norm(2).item() ** 2
        return total_norm_sq ** 0.5
    
    def training_step(self, batch, batch_idx):
        loss = 0.
        opt = self.optimizers()
        # opt.zero_grad()
        # random selection
        sch = self.lr_schedulers()
        x,y_gt,start_idx = batch
        y_pred = self.model(x,start_idx.to(x))
        
        if self.valid_indices is not None:  
            y_pred = y_pred[:, self.valid_indices] # ignore the last channel
            y_gt = y_gt[:, self.valid_indices]
        
        loss = self.criterion(y_pred, y_gt)
        self.manual_backward(loss)
        metric_dict = self.criterion.get_metrics(y_pred, y_gt, split='train', prefix=f'')
        grad_norm = self._compute_grad_norm()
        metric_dict[f'train/_grad_norm'] = grad_norm
        # breakpoint()
        # print(sch._step_count,sch.max_epochs)
        metric_dict[f'schedule_epoch'] = sch._step_count / sch.max_epochs * self.hparams.max_epochs
        self.log_dict(
            metric_dict,
            prog_bar=True,
            on_step=True,
            on_epoch=False,
            batch_size=y_pred.size(0),
        )
        #print(batch_idx,self.accumulate_grad_batches,self._nsteps)
        if (batch_idx + 1) % self.accumulate_grad_batches == 0:
            self.clip_gradients(opt, gradient_clip_val=10, gradient_clip_algorithm="norm")
            opt.step()
            opt.zero_grad()
            sch = self.lr_schedulers()
            sch.step()

    
    def on_validation_epoch_start(self):
        self.model.eval()
        opt = self.optimizers()
        try:
            opt.zero_grad()
        except:
            pass
        self.agg_loss = []
        self.agg_metrics = []
        
    def on_train_epoch_start(self):
        opt = self.optimizers()
        opt.zero_grad()
        self.model.train()
    
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        # breakpoint()
        x,y_gt,start_idx = batch
        y_pred = self.model(x,start_idx.to(x))

        # normalized metric dict
        # breakpoint()
        norm_metric_dict = self.criterion.get_metrics(y_pred, y_gt, split='val', prefix=f'_norm_', denormalize=False)
        # raw metric dict
        
        if self.valid_indices is not None:
            y_pred = y_pred[:, self.valid_indices] # ignore the last channel
            y_gt = y_gt[:, self.valid_indices]
        
        loss = self.criterion(y_pred, y_gt)
        self.agg_loss.append(loss.item())
        
        y_pred = self.trainer.datamodule.denormalize(y_pred, valid_indices=self.valid_indices)
        y_gt = self.trainer.datamodule.denormalize(y_gt,valid_indices=self.valid_indices)
        raw_metric_dict = self.criterion.get_metrics(y_pred, y_gt, split='val', prefix=f'_raw_', denormalize=False)
        metric_dict = {**norm_metric_dict, **raw_metric_dict,"start_idx":start_idx.item()}
        self.agg_metrics.append(metric_dict)
        
        # self.log_dict(
        #     metric_dict,
        #     prog_bar=True,
        #     on_step=False,
        #     on_epoch=True,
        #     batch_size=x.size(0),
        #     sync_dist=True,
        #     add_dataloader_idx=False,
        # )
        
        # # if first batch and gpu 0, save a sample pair
        # if batch_idx == 0 and self.global_rank == 0:
        #     self.sample_pair = (batch[0], output['reconstructions'][0])
        
    def process_agg_metrics(self,all_results):
        # Step 1: Group by start_idx
        all_data_prefix = []
        for row in all_results:
            all_data_prefix.append({f"{k}_{row['start_idx']}":v for k, v in row.items() if k != 'start_idx'})
        avg_data = pd.DataFrame(all_data_prefix).mean().to_dict()
        avg_dict_garher = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(avg_dict_garher, avg_data)
        all_data_across_gather = pd.DataFrame(avg_dict_garher).mean().to_dict()
        return all_data_across_gather
        

        
    @torch.no_grad()
    def on_validation_epoch_end(self):
        # aggregate self.agg_loss across all processes
        local_mean_loss = torch.tensor(sum(self.agg_loss) / len(self.agg_loss)).to(self.device)
        
        torch.distributed.barrier()
        processed_metrics = self.process_agg_metrics(self.agg_metrics)
        # if torch.distributed.get_rank() == 0:
        if self.trainer.global_rank == 0:
            self.log_dict(
                processed_metrics,
                prog_bar=False,
                on_step=False,
                on_epoch=True,
                sync_dist=False,
                rank_zero_only=True,
            )
        
        metrics_by_start_idx = {}
        for metric_dict in self.agg_metrics:
            start_idx = metric_dict['start_idx']
            if start_idx not in metrics_by_start_idx:
                metrics_by_start_idx[start_idx] = []
            metrics_by_start_idx[start_idx].append({k: v for k, v in metric_dict.items() if k != 'start_idx'})
        gathered_means = self.all_gather(local_mean_loss)
        # Calculate the global mean across all ranks
        global_mean_loss = gathered_means.mean().item()
        self.log(
            'val/loss',
            global_mean_loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        # if self.global_rank == 0 and self.trainer.current_epoch % self.hparams.visual_log_interval == 0:
            # original, reconstructed = self.sample_pair
            # combined = torch.stack([original, reconstructed], dim=0)
            # video_np = create_video_heatmap(combined.cpu().to(torch.float32).detach().numpy(), channel_names=self.trainer.datamodule.hparams.channel_names)
            # resized_video_np = resize_video_array(video_np, width=512)
            # resized_video_np = np.clip(np.expand_dims(resized_video_np.transpose(0, 3, 1, 2), axis=0).astype(np.uint8), 0, 255)
            # self.logger.experiment.log({f"media/comparison": wandb.Video(resized_video_np, fps=25, format="mp4")})
    
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
