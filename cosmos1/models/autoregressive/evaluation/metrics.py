# import torch
import numpy as np

def channel_wise_mse(pred, target):
    return np.mean((pred - target) ** 2, axis=(1, 2))

def mse(pred, target):
    return channel_wise_mse(pred, target).mean()

def psnr(pred, target):
    return 10 * np.log10(255 ** 2 / mse(pred, target))

def vrmse(pred, target):
    mean_prediction = target.mean(axis=(1, 2), keepdims=True)
    print("Mean Prediction Shape: ", mean_prediction.shape)
    channel_vrmse = channel_wise_mse(pred, target)/channel_wise_mse(mean_prediction, target)
    print("Channel VRMSE Shape: ", channel_vrmse.shape)
    return channel_vrmse.mean()


a = np.random.randn(10, 256, 256, 3)
b = np.random.randn(10, 256, 256, 3)

print("VRMSE Shape: ", vrmse(a, b).shape)