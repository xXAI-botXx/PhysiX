import numpy as np
import torch

from well_utils.metrics.common import Metric

class MSE(Metric):
    @staticmethod
    def eval(
        x: torch.Tensor | np.ndarray,
        y: torch.Tensor | np.ndarray,
        n_spatial_dims: int,
        eps: float = 1e-7,
    ) -> torch.Tensor:
        """
        Mean Squared Error

        Args:
            x: Input tensor.
            y: Target tensor.
            n_spatial_dims: Number of spatial dimensions.

        Returns:
            Mean squared error between x and y.
        """
        spatial_dims = tuple(range(-n_spatial_dims - 1, -1))
        return torch.mean((x - y) ** 2, dim=spatial_dims)

class NMSE(Metric):
    @staticmethod
    def eval(
        x: torch.Tensor | np.ndarray,
        y: torch.Tensor | np.ndarray,
        n_spatial_dims: int,
        eps: float = 1e-7,
        norm_mode: str = "norm",
    ) -> torch.Tensor:
        """
        Normalized Mean Squared Error

        Args:
            x: Input tensor.
            y: Target tensor.
            n_spatial_dims: Number of spatial dimensions.
            eps: Small value to avoid division by zero. Default is 1e-7.
            norm_mode: Mode for computing the normalization factor. Can be 'norm' or 'std'. Default is 'norm'.

        Returns:
            Normalized mean squared error between x and y.
        """
        spatial_dims = tuple(range(-n_spatial_dims - 1, -1))
        if norm_mode == "norm":
            norm = torch.mean(y**2, dim=spatial_dims)
        elif norm_mode == "std":
            norm = torch.std(y, dim=spatial_dims) ** 2
        else:
            raise ValueError(f"Invalid norm_mode: {norm_mode}")
        return MSE.eval(x, y, n_spatial_dims) / (norm + eps)

class RMSE(Metric):
    @staticmethod
    def eval(
        x: torch.Tensor | np.ndarray,
        y: torch.Tensor | np.ndarray,
        n_spatial_dims: int,
    ) -> torch.Tensor:
        """
        Root Mean Squared Error

        Args:
            x: Input tensor.
            y: Target tensor.
            n_spatial_dims: Number of spatial dimensions.

        Returns:
            Root mean squared error between x and y.
        """
        return torch.sqrt(MSE.eval(x, y, n_spatial_dims))

class NRMSE(Metric):
    @staticmethod
    def eval(
        x: torch.Tensor | np.ndarray,
        y: torch.Tensor | np.ndarray,
        n_spatial_dims: int,
        eps: float = 1e-7,
        norm_mode: str = "norm",
    ) -> torch.Tensor:
        """
        Normalized Root Mean Squared Error

        Args:
            x: Input tensor.
            y: Target tensor.
            n_spatial_dims: Number of spatial dimensions.
            eps: Small value to avoid division by zero. Default is 1e-7.
            norm_mode: Mode for computing the normalization factor. Can be 'norm' or 'std'. Default is 'norm'.

        Returns:
            Normalized root mean squared error between x and y.
        """
        return torch.sqrt(NMSE.eval(x, y, n_spatial_dims, eps=eps, norm_mode=norm_mode))

class VMSE(Metric):
    @staticmethod
    def eval(
        x: torch.Tensor | np.ndarray,
        y: torch.Tensor | np.ndarray,
        n_spatial_dims: int,
        eps: float = 1e-7,
    ) -> torch.Tensor:
        """
        Variance Scaled Mean Squared Error

        Args:
            x: Input tensor.
            y: Target tensor.
            n_spatial_dims: Number of spatial dimensions.

        Returns:
            Variance mean squared error between x and y.
        """
        return NMSE.eval(x, y, n_spatial_dims, norm_mode="std", eps=eps)

class VRMSE(Metric):
    @staticmethod
    def eval(
        x: torch.Tensor | np.ndarray,
        y: torch.Tensor | np.ndarray,
        n_spatial_dims: int,
        eps: float = 1e-7,
    ) -> torch.Tensor:
        """
        Root Variance Scaled Mean Squared Error

        Args:
            x: Input tensor.
            y: Target tensor.
            n_spatial_dims: Number of spatial dimensions.

        Returns:
            Root variance mean squared error between x and y.
        """
        return NRMSE.eval(x, y, n_spatial_dims, norm_mode="std", eps=eps)

class LInfinity(Metric):
    @staticmethod
    def eval(
        x: torch.Tensor | np.ndarray,
        y: torch.Tensor | np.ndarray,
        n_spatial_dims: int,
    ) -> torch.Tensor:
        """
        L-Infinity Norm

        Args:
            x: Input tensor.
            y: Target tensor.
            n_spatial_dims: Number of spatial dimensions.

        Returns:
            L-Infinity norm between x and y.
        """
        spatial_dims = tuple(range(-n_spatial_dims - 1, -1))
        return torch.max(
            torch.abs(x - y).flatten(start_dim=spatial_dims[0], end_dim=-2), dim=-1
        ).values