import torch
import time
import random

def duplicate_weights(tensor, input_channels, axis=1, duplicate_index=None):
    if axis not in [0, 1]:
        raise ValueError("Axis must be 0 or 1")

    # Get the size of the target dimension
    target_dim_size = tensor.size(axis)

    # Create a mask for every nth element in the target dimension
    mask = torch.arange(target_dim_size) % input_channels == duplicate_index

    # Create an index tensor for the new positions
    indices = torch.arange(target_dim_size)
    duplicate_indices = indices[mask]

    # Calculate new size for the target dimension
    new_dim_size = target_dim_size + duplicate_indices.size(0)
    new_shape = list(tensor.size())
    new_shape[axis] = new_dim_size
    result = torch.empty(new_shape, dtype=tensor.dtype, device=tensor.device)

    # Create a mask for where duplicates should go
    interleave_mask = (torch.arange(new_dim_size) % (input_channels + 1) == input_channels)[:new_dim_size]

    # Fill the result tensor along the specified axis
    if axis == 0:
        result[~interleave_mask, ...] = tensor
        result[interleave_mask, ...] = tensor[mask, ...]
    else:
        result[:, ~interleave_mask, ...] = tensor
        result[:, interleave_mask, ...] = tensor[:, mask, ...]

    return result

def remove_weights(tensor, current_channels, axis=1, remove_index=None):
    """Removes weights corresponding to a specific channel index."""
    if axis not in [0, 1]:
        raise ValueError("Axis must be 0 or 1")
    if remove_index is None or remove_index < 0 or remove_index >= current_channels:
        remove_index = random.randint(0, current_channels - 1) # Choose random index to remove

    target_dim_size = tensor.size(axis)
    if target_dim_size % current_channels != 0:
        # If the dimension size isn't divisible, something is wrong upstream or state is inconsistent.
        raise ValueError(f"Dimension {axis} size {target_dim_size} is not divisible by current_channels {current_channels}")

    # Create a mask for the indices to *keep*
    keep_mask = torch.ones(target_dim_size, dtype=torch.bool, device=tensor.device)
    # Identify indices to remove: start at remove_index, step by current_channels
    indices_to_remove = torch.arange(remove_index, target_dim_size, current_channels, device=tensor.device)
    keep_mask[indices_to_remove] = False

    # Select the elements to keep along the specified axis
    if axis == 0:
        result = tensor[keep_mask, ...]
    else: # axis == 1
        result = tensor[:, keep_mask, ...]

    return result

def inflate_channel_weights(weights, input_channels, output_channels, *, modify_input_shape=False, modify_output_shape=False):
    if output_channels > input_channels:
        # Inflation logic (increase channels)
        for channels in range(input_channels, output_channels):
            duplicate_index = random.randint(0, channels - 1)
            if modify_input_shape:
                weights = duplicate_weights(weights, channels, axis=1, duplicate_index=duplicate_index)
            if modify_output_shape:
                # If both input and output are modified, the number of output channels for duplication
                # should match the current state of the weights tensor's axis 0.
                current_output_channels = weights.shape[0] // (weights.shape[0] // channels) if modify_input_shape else channels
                weights = duplicate_weights(weights, current_output_channels, axis=0, duplicate_index=duplicate_index)

    elif output_channels < input_channels:
        # Deflation logic (decrease channels)
        for channels in range(input_channels, output_channels, -1): # Iterate downwards from current to target
            remove_index = random.randint(0, channels - 1) # Select a random channel index to remove from the current set
            if modify_input_shape:
                weights = remove_weights(weights, channels, axis=1, remove_index=remove_index)
            if modify_output_shape:
                # Similar to inflation, ensure the 'channels' count matches the current state of axis 0
                # before removal.
                current_output_channels = weights.shape[0] // (weights.shape[0] // channels) if modify_input_shape else channels
                weights = remove_weights(weights, current_output_channels, axis=0, remove_index=remove_index)

    # If input_channels == output_channels, do nothing
    return weights

# start = time.time()
# duplicated = duplicate_weights(torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), 10)
# print(time.time() - start)

# print(duplicated)

"""
python -m cosmos1.models.autoregressive.tokenizer.lobotomize.helpers
"""