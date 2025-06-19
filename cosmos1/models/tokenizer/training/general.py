from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, Tuple, Union
import numpy as np
import torch
import wandb
import shutil
import os
import torch.distributed as dist
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from collections import defaultdict
import einops

from focal_frequency_loss import FocalFrequencyLoss as FFL

from cosmos1.models.autoregressive.tokenizer.training.data_loader import RandomHDF5Dataset
from cosmos1.models.autoregressive.tokenizer.networks import NetworkEval

from well_utils.data_processing.visualizations import create_video_heatmap
from well_utils.data_processing.helpers import resize_video_array
from well_utils.metrics.spatial import VRMSE
from well_utils.data_processing.normalization.torch_normalize import TorchNormalizationApplier

NUM_CONTEXT_FRAMES = 33
CHANNEL_NAMES = ['tracer', 'pressure', 'velocity_x', 'velocity_y']

def log_video_comparison(original: Tensor, reconstructed: Tensor, prefix: str = "val") -> None:
    combined = torch.stack([original, reconstructed], dim=0)
    video_np = create_video_heatmap(combined.cpu().to(torch.float32).detach().numpy()/2, fps=25, channel_names=CHANNEL_NAMES)
    resized_video_np = resize_video_array(video_np, width=512)
    resized_video_np = np.expand_dims(resized_video_np.transpose(0, 3, 1, 2), axis=0)
    wandb.log({f"{prefix}_comparison": wandb.Video(resized_video_np, fps=25, format="mp4")})

class VAELoss(nn.Module):
    def __init__(self, beta=1.0, recon_loss='l1'):
        super().__init__()
        self.beta = beta
        if recon_loss == 'l1':
            self.recon_criterion = nn.L1Loss()
        elif recon_loss == 'mse':
            self.recon_criterion = nn.MSELoss()
        elif recon_loss == 'vrmse':
            self.recon_criterion = VRMSE(n_spatial_dims=2, reduce=True, eps=0.01)
        else:
            raise ValueError(f"Unsupported reconstruction loss: {recon_loss}")
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
        self.vrmse = VRMSE(n_spatial_dims=2, reduce=True, eps=0.01)

    def forward(self, output, target):
        recon_loss = self.recon_criterion(output['reconstructions'], target)
        mean, log_var = output['posteriors']
        # kl_loss = 0.5 * torch.sum(mean.pow(2) + log_var.exp() - log_var - 1, dim=[1, 2, 3, 4])
        # kl_loss = torch.mean(kl_loss)
        kl_loss = torch.tensor(0.0)
        total_loss = recon_loss# + self.beta * kl_loss
        return total_loss, recon_loss, kl_loss

class VAETrainer:
    def __init__(self, autoencoder_path: Path, device: torch.device, grad_accumulation_steps: int = 1,
                 clip_grad_norm: float = 1.0, dtype: torch.dtype = torch.float32, stats_path: Path = None,
                 beta: float = 1.0, recon_loss_type: str = 'l1'):
        self.device = device
        self.grad_accumulation_steps = grad_accumulation_steps
        self._accumulation_counter = 0
        self.clip_grad_norm = clip_grad_norm
        self.dtype = dtype
        self.model = self._init_model(autoencoder_path)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        self.criterion = VAELoss(beta=beta, recon_loss=recon_loss_type)
        self.metrics = {'VRMSE': VRMSE(n_spatial_dims=2, reduce=True), 
                        'MSE': nn.MSELoss(), 
                        'L1': nn.L1Loss()}
        self.normalizer = TorchNormalizationApplier(stats_path=stats_path).to(self.device, dtype=self.dtype) if stats_path is not None else lambda x: x

    def _init_model(self, autoencoder_path) -> nn.Module:
        model = torch.load(autoencoder_path, weights_only=False)
        model = model.to(self.device, dtype=self.dtype)
        if dist.is_initialized() and dist.get_world_size() > 1:
            model = DDP(model, device_ids=[self.device.index], output_device=self.device.index)
        return model

    def print_model_summary(self):
        for name, param in self.model.named_parameters():
            print(name, param.shape, param.requires_grad, param.grad.abs().mean().item() if param.grad is not None else None)

    def _compute_grad_norm(self):
        total_norm_sq = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                total_norm_sq += p.grad.data.norm(2).item() ** 2
        grad_norm = total_norm_sq ** 0.5
        if dist.get_rank() == 0:
            wandb.log({"grad_norm": grad_norm})
        return grad_norm

    def train_step(self, batch: Tensor) -> Dict[str, float]:
        self.model.train()
        if self._accumulation_counter == 0:
            self.optimizer.zero_grad()
        batch = batch.to(self.device)
        output = self.model(batch)
        total_loss, recon_loss, kl_loss = self.criterion(output, batch)
        (total_loss / self.grad_accumulation_steps).backward()
        self._accumulation_counter += 1
        metrics = self._compute_metrics(batch, output, 'train')
        metrics['train_total_loss'] = total_loss.item()
        metrics['train_recon_loss'] = recon_loss.item()
        metrics['train_kl_loss'] = kl_loss.item()
        if self._accumulation_counter == self.grad_accumulation_steps:
            self._compute_grad_norm()
            model_to_clip = self.model.module if isinstance(self.model, DDP) else self.model
            torch.nn.utils.clip_grad_norm_(model_to_clip.parameters(), self.clip_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()
            self._accumulation_counter = 0
        return metrics

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Tuple[Dict[str, float], Tuple[Tensor, Tensor]]:
        self.model.eval()
        metrics = defaultdict(float)
        sample_pair = (None, None)
        total_samples = 0
        for i, batch in enumerate(val_loader):
            batch = batch.to(self.device)
            output = self.model(batch)
            output = output._asdict()
            batch_metrics = self._compute_metrics(batch, output, 'val')
            batch_size = batch.size(0)
            for k, v in batch_metrics.items():
                metrics[k] += v * batch_size
            total_samples += batch_size
            if i == 0 and dist.get_rank() == 0:
                sample_pair = (batch[0], output['reconstructions'][0])
        for k in list(metrics.keys()):
            tensor = torch.tensor(metrics[k], device=self.device)
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            metrics[k] = tensor.item() / (total_samples * dist.get_world_size())
        return dict(metrics), sample_pair

    def _compute_metrics(self, original: Tensor, output: Dict, prefix: str) -> Dict[str, float]:
        total_loss, recon_loss, kl_loss = self.criterion(output, original)
        # Compute and report spatial metrics.
        metrics = {
            f'{prefix}_recon_loss': recon_loss,
            **{f'{prefix}_{name}': function(self.normalizer.inverse(original),
                                             self.normalizer.inverse(output['reconstructions'])).item()
               for name, function in self.metrics.items()}
        }
        metrics[f'{prefix}_total_loss'] = total_loss.item()
        metrics[f'{prefix}_kl_loss'] = kl_loss.item()
        return metrics

    def save_checkpoint(self, checkpoint_dir: Path, epoch: int, checkpoint_type: str):
        if dist.get_rank() != 0:
            return
        if checkpoint_type == "best":
            file_name = "best_model.pt"
        elif checkpoint_type == "recent":
            file_name = f"recent_model_epoch_{epoch}.pt"
        else:
            raise ValueError(f"Invalid checkpoint type: {checkpoint_type}")
        path = checkpoint_dir / file_name
        if checkpoint_type == "best" and path.exists():
            path.unlink()
        if checkpoint_type == "recent":
            for p in checkpoint_dir.glob("recent_model_epoch_*.pt"):
                if p.is_file():
                    p.unlink()
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        model_to_save = self.model.module if isinstance(self.model, DDP) else self.model
        torch.save(model_to_save, path)
        print(f"Saved {checkpoint_type} model checkpoint at {path}")

def initial_validation(trainer, val_loader):
    val_metrics, (val_samples, val_reconstructions) = trainer.validate(val_loader)
    if dist.get_rank() == 0:
        log_video_comparison(val_samples, val_reconstructions, prefix="val")
        wandb.log(val_metrics)
        print(f"Initial validation loss: {val_metrics['val_total_loss']:.4f}")
    return val_metrics['val_total_loss']

def main(args):
    dist.init_process_group(backend='nccl', init_method='env://')
    args.local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(args.local_rank)
    device = torch.device(f'cuda:{args.local_rank}')
    if dist.get_rank() == 0:
        wandb.init(config=vars(args))
    else:
        os.environ['WANDB_MODE'] = 'disabled'
        wandb.init(mode="disabled")
    train_dataset = RandomHDF5Dataset(
        data_dir=args.train_data_path,
        dtype=torch.bfloat16,
        n_frames=NUM_CONTEXT_FRAMES,
        data_resolution=args.data_resolution,
    )
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    val_dataset = RandomHDF5Dataset(
        data_dir=args.val_data_path,
        dtype=torch.bfloat16,
        n_frames=NUM_CONTEXT_FRAMES,
        data_resolution=args.data_resolution,
    )
    val_sampler = DistributedSampler(val_dataset, shuffle=False)
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True
    )
    trainer = VAETrainer(
        autoencoder_path=args.autoencoder_path,
        device=device,
        grad_accumulation_steps=args.grad_accumulation_steps,
        clip_grad_norm=args.clip_grad_norm,
        dtype=torch.bfloat16,
        stats_path=Path(args.stats_path),
        beta=args.beta,
        recon_loss_type=args.recon_loss_type
    )
    best_val_loss = initial_validation(trainer, val_loader)
    for epoch in range(args.epochs):
        train_loader.sampler.set_epoch(epoch)
        epoch_metrics = defaultdict(float)
        total_train_samples = 0
        for batch in train_loader:
            batch_size = batch.size(0)
            total_train_samples += batch_size
            batch_metrics = trainer.train_step(batch)
            for k, v in batch_metrics.items():
                epoch_metrics[k] += v * batch_size
        if trainer._accumulation_counter > 0:
            trainer._compute_grad_norm()
            model_to_clip = trainer.model.module if isinstance(trainer.model, DDP) else trainer.model
            torch.nn.utils.clip_grad_norm_(model_to_clip.parameters(), trainer.clip_grad_norm)
            trainer.optimizer.step()
            trainer.optimizer.zero_grad()
            trainer._accumulation_counter = 0
        total_train_samples_tensor = torch.tensor(total_train_samples, device=device)
        dist.all_reduce(total_train_samples_tensor, op=dist.ReduceOp.SUM)
        for k in list(epoch_metrics.keys()):
            tensor = torch.tensor(epoch_metrics[k], device=device)
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            epoch_metrics[k] = tensor.item() / total_train_samples_tensor.item()
        val_metrics, (val_samples, val_reconstructions) = trainer.validate(val_loader)
        current_val_loss = val_metrics['val_total_loss']
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            trainer.save_checkpoint(args.checkpoint_dir, epoch+1, "best")
        if (epoch + 1) % args.save_every_n_epochs == 0:
            trainer.save_checkpoint(args.checkpoint_dir, epoch+1, "recent")
        if dist.get_rank() == 0:
            metrics = {"epoch": epoch, **epoch_metrics, **val_metrics}
            wandb.log(metrics)
            if (epoch + 1) % args.visual_log_interval == 0:
                log_video_comparison(val_samples, val_reconstructions)
            print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {metrics['train_total_loss']:.4f} | Val Loss: {metrics['val_total_loss']:.4f}")
    if dist.get_rank() == 0:
        wandb.finish()
    dist.destroy_process_group()

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--train_data_path", type=Path, required=True)
    parser.add_argument("--val_data_path", type=Path, required=True)
    parser.add_argument("--autoencoder_path", type=Path, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--checkpoint_dir", type=Path, required=True)
    parser.add_argument("--save_every_n_epochs", type=int, required=True)
    parser.add_argument("--visual_log_interval", type=int, default=10)
    parser.add_argument("--data_resolution", nargs=2, type=int, default=None)
    parser.add_argument("--grad_accumulation_steps", type=int, default=1)
    parser.add_argument("--clip_grad_norm", type=float, default=1.0)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--stats_path", type=Path, default=None)
    parser.add_argument("--beta", type=float, default=1.0, help="Weight for KL divergence loss")
    parser.add_argument("--recon_loss_type", type=str, default='l1', choices=['l1', 'mse', 'vrmse'])
    return parser.parse_args()

if __name__ == "__main__":
    main(parse_args())