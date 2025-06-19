def mse(ground_truth, predictions):
    return ((ground_truth - predictions) ** 2).mean().item()

import numpy as np

def compute_vrmse(ground_truth, predicted, time_window):
    """
    Compute the VRMSE metric for a given time window.
    
    Parameters:
    ground_truth (numpy.ndarray): Ground truth video data (time, height, width)
    predicted (numpy.ndarray): Predicted video data (time, height, width)
    time_window (tuple): Start and end of time window (inclusive)
    
    Returns:
    float: VRMSE value for the specified time window
    """
    start, end = time_window
    
    ground_truth_window = ground_truth[start:end+1]
    predicted_window = predicted[start:end+1]
    
    mean_value = np.mean(ground_truth_window)
    
    mse = np.mean((ground_truth_window - predicted_window) ** 2)
    rmse = np.sqrt(mse)
    
    vrmse = rmse / mean_value
    
    return vrmse


def calculate_metrics(ground_truth, predictions):
    if ground_truth.shape != predictions.shape:
        raise ValueError(f"Ground truth ({ground_truth.shape}) and predictions must have the same shape ({predictions.shape})")

    return {"mse": mse(ground_truth, predictions),
            "vrmse": compute_vrmse(ground_truth, predictions, (0, ground_truth.shape[0]-1))}