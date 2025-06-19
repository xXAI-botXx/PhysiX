import torch
import torch.nn as nn
from typing import List
from well_utils.metrics.spatial import VRMSE
from well_utils.data_processing.normalization.torch_normalize import TorchNormalizationApplier

class NL1Loss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, output, target):
        squared_error = torch.abs(output - target)
        spatial_std = torch.std(target, dim=[-2, -1], keepdim=True)
        normalized_squared_error = squared_error / (spatial_std + self.eps)
        return torch.mean(normalized_squared_error)


class BaseVAELoss(nn.Module):
    def __init__(self, recon_loss_type: str = 'l1', normalization_path: str = None, constant_channels: List[int] = [], **kwargs):
        super().__init__()
        self.normalizer = TorchNormalizationApplier(normalization_path, normalization_type='standard', constant_channels=constant_channels) if normalization_path else None
        self.metrics = {'VRMSE': VRMSE(n_spatial_dims=2, reduce=True), 'MSE': torch.nn.MSELoss(), 'L1': torch.nn.L1Loss()}
        self.constant_channels = constant_channels if constant_channels is not None else []
        if recon_loss_type == 'l1':
            self.recon_loss_fn = nn.L1Loss()
        elif recon_loss_type == 'mse':
            self.recon_loss_fn = nn.MSELoss()
        elif recon_loss_type == 'vrmse':
            self.recon_loss_fn = VRMSE(n_spatial_dims=2, reduce=True)
        elif recon_loss_type == 'nl1':
            self.recon_loss_fn = NL1Loss(eps=1e-3)
        else:
            raise ValueError(f"Unsupported reconstruction loss type: {recon_loss_type}")

    def _filter_constant_channels(self, target):
        """Remove constant channels from target tensor to match output tensor which already excludes them."""
        if not self.constant_channels:
            return target
            
        valid_channels = [i for i in range(target.shape[1]) if i not in self.constant_channels]
            
        return target[:, valid_channels]

    def forward(self, output, target):
        raise NotImplementedError("Subclasses must implement forward()")
    
    @torch.no_grad()
    def get_metrics(self, output, target, split: str = '', prefix: str = '', denormalize: bool = False):
        if isinstance(output, dict):
            reconstructions = output['reconstructions']
        elif isinstance(output, torch.Tensor):
            reconstructions = output
        else:
            raise ValueError(f"Output must be a tensor or a dictionary containing 'reconstructions' key., got: {type(output)}")
        if denormalize:
            reconstructions = self.normalizer.inverse_norm(reconstructions, ignore_constant_channels=True)
            target = self.normalizer.inverse_norm(target)
            
        # Filter out constant channels for metrics
        filtered_target = self._filter_constant_channels(target)
        
        recon_loss = self.recon_loss_fn(reconstructions, filtered_target).item()
        return {f'{split}/{prefix}recon_loss': recon_loss, 
                **{f'{split}/{prefix}{name}': function(reconstructions, filtered_target).item() 
                   for name, function in self.metrics.items()}}


class DiscreteVAELoss(BaseVAELoss):
    def forward(self, output, target):
        filtered_target = self._filter_constant_channels(target)
        reconst = output['reconstructions'] if isinstance(output, dict) else output
        return self.recon_loss_fn(reconst, filtered_target)


class ContinuousVAELoss(BaseVAELoss):
    def __init__(self, beta: float = 1.0, formulation: str = 'VAE', recon_loss_type: str = 'l1', **kwargs):
        super().__init__(recon_loss_type, **kwargs)
        self.beta = beta
        self.formulation = formulation
    
    def forward(self, output, target):
        filtered_target = self._filter_constant_channels(target)
        recon_loss = self.recon_loss_fn(output['reconstructions'], filtered_target)
        mean, log_var = output['posteriors']
        if self.formulation == 'VAE':
            kl_loss = 0.5 * torch.mean(
                torch.sum(torch.exp(log_var) + mean.pow(2) - 1 - log_var, dim=[1, 2, 3, 4])
            )
        else:
            kl_loss = 0.0
        return recon_loss + self.beta * kl_loss
    
    def get_metrics(self, output, target, split: str = '', prefix: str = '', denormalize: bool = False):
        metric_vals = super().get_metrics(output, target, split, prefix, denormalize)
        recon_loss = self.recon_loss_fn(output['reconstructions'], target).item()
        mean, log_var = output['posteriors']
        if self.formulation == 'VAE':
            kl_loss = 0.5 * torch.mean(
                torch.sum(torch.exp(log_var) + mean.pow(2) - 1 - log_var, dim=[1, 2, 3, 4])
            ).item()
            metric_vals[f'{split}/{prefix}kl_loss'] = kl_loss
            metric_vals[f'{split}/{prefix}total_loss'] = recon_loss + self.beta * kl_loss

        return metric_vals


def build_vae_loss(mode: str, beta: float = 1.0, recon_loss_type: str = 'l1', normalization_path: str = None, constant_channels: List[int] = [], **kwargs) -> BaseVAELoss:
    print("Constant Channels", constant_channels)
    if mode == 'continuous':
        return ContinuousVAELoss(beta=beta, recon_loss_type=recon_loss_type, normalization_path=normalization_path, constant_channels=constant_channels, **kwargs)
    elif mode == 'discrete':
        return DiscreteVAELoss(recon_loss_type=recon_loss_type, normalization_path=normalization_path, constant_channels=constant_channels, **kwargs)
    else:
        raise ValueError(f"Invalid mode: {mode}") 