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

# Configuration
NUM_CONTEXT_FRAMES = 33
CHANNEL_NAMES = ['buoyancy', 'pressure', 'velocity_x', 'velocity_y']

def log_video_comparison(original: Tensor, reconstructed: Tensor, prefix: str = "val") -> None:
    combined = torch.stack([original, reconstructed], dim=0)
    video_np = create_video_heatmap(combined.cpu().to(torch.float32).detach().numpy()/2, fps=25, channel_names=CHANNEL_NAMES)
    resized_video_np = resize_video_array(video_np, width=512)
    resized_video_np = np.expand_dims(resized_video_np.transpose(0, 3, 1, 2), axis=0)
    wandb.log({f"{prefix}_comparison": wandb.Video(resized_video_np, fps=25, format="mp4")})

class CustomLoss(nn.Module):
    def __init__(self, loss='l1'):
        super().__init__()
        self.vrmse = VRMSE(n_spatial_dims=2, reduce=True, eps=.01)
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
        self.ffl = FFL(alpha=1)
        self.loss = loss

    def flatten_time(self, x):
        return einops.rearrange(x, 'b c t h w -> (b t) c h w')
    
    def unflatten_time(self, x):
        return einops.rearrange(x, '(b t) c h w -> b c t h w', t=NUM_CONTEXT_FRAMES)
    
    def forward(self, input, target):
        original_type = input.dtype
        if self.loss == 'l1':
            return self.l1(input, target)# + (.1*self.ffl(self.flatten_time(input).to(torch.float32), self.flatten_time(target).to(torch.float32)).to(original_type))
        elif self.loss == 'mse':
            return self.mse(input, target)
        elif self.loss == 'vrmse':
            return self.vrmse(input, target)
        else:
            raise ValueError(f"Invalid loss function: {self.loss}")

class VideoAutoencoderTrainer:
    def __init__(self, autoencoder_path: Path, device: torch.device, grad_accumulation_steps: int = 1, clip_grad_norm: float = 1.0):
        self.device = device
        self.grad_accumulation_steps = grad_accumulation_steps
        self._accumulation_counter = 0
        self.clip_grad_norm = clip_grad_norm
        self.model = self._init_model(autoencoder_path)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        self.criterion = CustomLoss()
        self.metrics = {'VRMSE': VRMSE(n_spatial_dims=2, reduce=True), 'MSE': nn.MSELoss(), 'L1': nn.L1Loss()}

    def _init_model(self, autoencoder_path) -> nn.Module:
        model = torch.load(autoencoder_path, weights_only=False).to(self.device)
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
    
    def _compute_unique_tokens(self, output: NetworkEval) -> Tensor:
        quant_info = output.quant_info.flatten()
        unique_tokens = len(torch.unique(quant_info))
        return unique_tokens

    def train_step(self, batch: Tensor) -> Dict[str, float]:
        self.model.train()
        # Zero gradients if starting a new accumulation cycle
        if self._accumulation_counter == 0:
            self.optimizer.zero_grad()

        batch = batch.to(self.device)
        output = self.model(batch)
        reconstructed = output['reconstructions']
        recon_loss = self.criterion(reconstructed, batch)
        # Scale loss for gradient accumulation before backward
        (recon_loss / self.grad_accumulation_steps).backward()
        self._accumulation_counter += 1

        # Compute metrics based on the current batch (unscaled)
        metrics = self._compute_metrics(batch, output, 'train')

        # When enough mini-batches are accumulated, perform an optimizer step
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
            batch_metrics = self._compute_metrics(batch, output, 'val')
            batch_size = batch.size(0)
            for k, v in batch_metrics.items():
                metrics[k] += v * batch_size
            total_samples += batch_size
            if i == 0 and dist.get_rank() == 0:
                sample_pair = (batch[0], output.reconstructions[0])

        # Aggregate metrics across all processes
        for k in list(metrics.keys()):
            tensor = torch.tensor(metrics[k], device=self.device)
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            metrics[k] = tensor.item() / (total_samples * dist.get_world_size())  # total_samples is per process
        metrics['unique_tokens'] = self._compute_unique_tokens(output)
        return dict(metrics), sample_pair

    def _compute_metrics(self, original: Tensor, output: Union[Dict, NetworkEval], prefix: str) -> Dict[str, float]:
        if isinstance(output, dict):
            reconstructed = output['reconstructions']
        else:
            reconstructed = output.reconstructions
            quant_loss = output.quant_loss
        recon_loss = self.criterion(reconstructed, original).item()
        return {
            f'{prefix}_recon_loss': recon_loss,
            **{f'{prefix}_{name}': function(original, reconstructed).item() for name, function in self.metrics.items()}
        }

    def save_checkpoint(self, checkpoint_dir: Path, epoch: int, checkpoint_type: str):
        if dist.get_rank() != 0:
            return  # Only save on rank 0

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
        print(f"Initial validation loss: {val_metrics['val_recon_loss']:.4f}")
    return val_metrics['val_recon_loss']

def main(args):
    # Initialize distributed training
    dist.init_process_group(backend='nccl', init_method='env://')
    args.local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(args.local_rank)
    device = torch.device(f'cuda:{args.local_rank}')
    
    # Setup wandb only on rank 0
    if dist.get_rank() == 0:
        wandb.init(config=vars(args))
    else:
        os.environ['WANDB_MODE'] = 'disabled'
        wandb.init(mode="disabled")
    
    # Create data loaders with DistributedSampler
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

    trainer = VideoAutoencoderTrainer(
        autoencoder_path=args.autoencoder_path,
        device=device,
        grad_accumulation_steps=args.grad_accumulation_steps,
        clip_grad_norm=args.clip_grad_norm
    )
    
    best_val_loss = initial_validation(trainer, val_loader)
    
    for epoch in range(args.epochs):
        train_loader.sampler.set_epoch(epoch)
        epoch_metrics = defaultdict(float)
        total_train_samples = 0  # Accumulate total samples seen this epoch
        for batch in train_loader:
            batch_size = batch.size(0)
            total_train_samples += batch_size
            batch_metrics = trainer.train_step(batch)
            for k, v in batch_metrics.items():
                epoch_metrics[k] += v * batch_size

        # If there are leftover accumulated gradients, perform an optimizer step
        if trainer._accumulation_counter > 0:
            trainer._compute_grad_norm()
            model_to_clip = trainer.model.module if isinstance(trainer.model, DDP) else trainer.model
            torch.nn.utils.clip_grad_norm_(model_to_clip.parameters(), trainer.clip_grad_norm)
            trainer.optimizer.step()
            trainer.optimizer.zero_grad()
            trainer._accumulation_counter = 0

        # Aggregate training metrics across distributed processes
        total_train_samples_tensor = torch.tensor(total_train_samples, device=device)
        dist.all_reduce(total_train_samples_tensor, op=dist.ReduceOp.SUM)
        for k in list(epoch_metrics.keys()):
            tensor = torch.tensor(epoch_metrics[k], device=device)
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            epoch_metrics[k] = tensor.item() / total_train_samples_tensor.item()

        val_metrics, (val_samples, val_reconstructions) = trainer.validate(val_loader)
        current_val_loss = val_metrics['val_recon_loss']
        
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
            print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {metrics['train_recon_loss']:.4f} | Val Loss: {metrics['val_recon_loss']:.4f}")

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
    parser.add_argument("--grad_accumulation_steps", type=int, default=1, help="Number of steps for gradient accumulation")
    parser.add_argument("--clip_grad_norm", type=float, default=1.0, help="Max norm for gradient clipping")
    parser.add_argument("--local_rank", type=int, default=0)  # Automatically passed by torchrun
    return parser.parse_args()

if __name__ == "__main__":
    main(parse_args())


"""
torchrun --nproc_per_node=$NUM_GPUS -m cosmos1.models.autoregressive.tokenizer.training.general \
    --train_data_path /data0/arshkon/data/the_well/normalized/shear_flow_4c/train \
    --val_data_path /data0/arshkon/data/the_well/normalized/shear_flow_4c/valid \
    --autoencoder_path /data0/arshkon/checkpoints/cosmos/Cosmos-1.0-Tokenizer-DV8x16x16/autoencoder_4c_8x8x8.pt \
    --checkpoint_dir /data0/arshkon/checkpoints/cosmos/finetuned/tokenizers/shear_flow_8x \
    --batch_size 1 \
    --epochs 5000 \
    --save_every_n_epochs 5 \
    --visual_log_interval 5 \
    --data_resolution 256 512 \
    --grad_accumulation_steps 8 \
    --clip_grad_norm 1.0
"""