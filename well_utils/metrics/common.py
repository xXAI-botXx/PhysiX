import numpy as np
import torch
import torch.nn as nn


class Metric(nn.Module):
    """
    Decorator for metrics that standardizes the input arguments and checks the dimensions of the input tensors.

    Args:
        f: function
            Metric function that takes in the following arguments:
            x: torch.Tensor | np.ndarray
                Input tensor.
            y: torch.Tensor | np.ndarray
                Target tensor.
            meta: WellMetadata
                Metadata for the dataset.
            **kwargs : dict
                Additional arguments for the metric.
    """
    def __init__(self, n_spatial_dims: int = None, reduce: str = None, eps: float = 1e-7):
        super(Metric, self).__init__()
        self.n_spatial_dims = n_spatial_dims
        self.eps = eps
        self.reduce = reduce

    def forward(self, *args, **kwargs):
        if self.n_spatial_dims is None:
            assert len(args) >= 3, "At least three arguments required (x, y, and n_spatial_dims)"
            x, y, n_spatial_dims = args[:3]
        else:
            assert len(args) >= 2, "At least two arguments required (x, y)"
            x, y = args[:2]
            n_spatial_dims = self.n_spatial_dims

        # Convert x and y to torch.Tensor if they are np.ndarray
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y)
        assert isinstance(x, torch.Tensor), "x must be a torch.Tensor or np.ndarray"
        assert isinstance(y, torch.Tensor), "y must be a torch.Tensor or np.ndarray"

        x = x.moveaxis(-4, -1)
        y = y.moveaxis(-4, -1)

        # Check dimensions
        assert (
            x.ndim >= n_spatial_dims + 1
        ), "x must have at least n_spatial_dims + 1 dimensions"
        assert (
            y.ndim >= n_spatial_dims + 1
        ), "y must have at least n_spatial_dims + 1 dimensions"

        metric_value = self.eval(x, y, n_spatial_dims, eps=self.eps, **kwargs)

        if self.reduce is None:
            return metric_value
        else:
            return metric_value.mean()

    @staticmethod
    def eval(self, x, y, n_spatial_dims, eps=1e-7, **kwargs):
        raise NotImplementedError
