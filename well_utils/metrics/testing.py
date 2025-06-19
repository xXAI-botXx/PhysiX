import torch

from well_utils.metrics.spatial import VMSE

metric = VMSE()

def test_nmse():
    x = torch.randn(1, 5, 3, 10, 10)
    y = torch.randn(1, 5, 3, 10, 10)
    y[:, :, 0] = 0
    result = metric(x, y, 2)
    print(result)
    print(result.shape)


if __name__ == "__main__":
    test_nmse()


"""
python -m well_utils.metrics.testing
"""
