import argparse
import numpy as np
import torch
from pathlib import Path
import h5py
import tempfile
import json
import time
import matplotlib.pyplot as plt

from well_utils.data_processing.normalization.normalize import NormalizationApplier
from well_utils.data_processing.normalization.torch_normalize import TorchNormalizationApplier

def create_test_stats_file(temp_dir, n_channels=3):
    """Create a temporary stats file with known values for testing"""
    stats = []
    for i in range(n_channels):
        stats.append({
            "min": -1.0 - i,  # Different stats for each channel
            "max": 1.0 + i,
            "mean": 0.1 * i,
            "std": 0.5 + 0.1 * i
        })
    
    stats_path = Path(temp_dir) / "test_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f)
    
    return stats_path

def create_sample_data(shape, seed=42):
    """Create sample data for testing"""
    np.random.seed(seed)
    # Generate random data
    numpy_data = np.random.randn(*shape).astype(np.float32)
    
    # Create equivalent torch tensor
    torch_data = torch.from_numpy(numpy_data).float()
    
    return numpy_data, torch_data

def verify_manually(data, stats, norm_type):
    """Manually verify normalization for a few points"""
    if norm_type == "standard":
        # For standard normalization: (x - mean) / std
        for c in range(len(stats)):
            if data.ndim == 4:  # (C, T, H, W)
                x = data[c, 0, 0, 0]
            else:  # (B, C, T, H, W)
                x = data[0, c, 0, 0, 0]
            mean = stats[c]["mean"]
            std = stats[c]["std"]
            expected = (x - mean) / (std + 1e-7)
            return expected
    else:  # minmax
        # For minmax normalization: 2 * (x - min) / (max - min) - 1
        for c in range(len(stats)):
            if data.ndim == 4:  # (C, T, H, W)
                x = data[c, 0, 0, 0]
            else:  # (B, C, T, H, W)
                x = data[0, c, 0, 0, 0]
            min_val = stats[c]["min"]
            max_val = stats[c]["max"]
            expected = 2 * (x - min_val) / (max_val - min_val + 1e-7) - 1
            return expected

def test_normalization_equality(stats_path, norm_type='standard'):
    """Test if both normalizers produce the same results"""
    print(f"\n=== Testing {norm_type} normalization equality ===")
    
    # Load stats for verification
    with open(stats_path, 'r') as f:
        stats = json.load(f)
    
    # Initialize normalizers
    numpy_normalizer = NormalizationApplier(stats_path, normalization_type=norm_type)
    torch_normalizer = TorchNormalizationApplier(stats_path, normalization_type=norm_type)
    
    # Get number of channels from stats
    n_channels = numpy_normalizer.n_channels
    
    # Test shapes
    shapes = [
        # (C, T, H, W) - no batch
        (n_channels, 10, 16, 16),
        # (B, C, T, H, W) - with batch
        (5, n_channels, 10, 16, 16)
    ]
    
    for shape in shapes:
        print(f"\nTesting shape: {shape}")
        numpy_data, torch_data = create_sample_data(shape)
        
        # For numpy normalizer, channels must be in the last dimension
        # For (C, T, H, W) -> (T, H, W, C)
        # For (B, C, T, H, W) -> (B, T, H, W, C)
        if len(shape) == 4:  # (C, T, H, W)
            numpy_data_for_normalizer = numpy_data.transpose(1, 2, 3, 0)
        else:  # (B, C, T, H, W)
            numpy_data_for_normalizer = numpy_data.transpose(0, 2, 3, 4, 1)
        
        # Time the operations
        start_time = time.time()
        numpy_norm = numpy_normalizer.normalize(numpy_data_for_normalizer)
        numpy_time = time.time() - start_time
        
        # For result comparison, transpose back to match torch format
        if len(shape) == 4:  # (T, H, W, C) -> (C, T, H, W)
            numpy_norm_for_comparison = numpy_norm.transpose(3, 0, 1, 2)
        else:  # (B, T, H, W, C) -> (B, C, T, H, W)
            numpy_norm_for_comparison = numpy_norm.transpose(0, 4, 1, 2, 3)
        
        # For torch tensor
        start_time = time.time()
        with torch.no_grad():
            torch_norm = torch_normalizer(torch_data).numpy()
        torch_time = time.time() - start_time
        
        # Calculate difference
        max_diff = np.max(np.abs(numpy_norm_for_comparison - torch_norm))
        print(f"Normalization max difference: {max_diff:.8f}")
        print(f"Numpy time: {numpy_time:.6f}s, Torch time: {torch_time:.6f}s")
        
        # Manually verify a few points
        expected_value = verify_manually(numpy_data, stats, norm_type)
        if len(shape) == 4:
            actual_numpy = numpy_norm_for_comparison[0, 0, 0, 0]
            actual_torch = torch_norm[0, 0, 0, 0]
        else:
            actual_numpy = numpy_norm_for_comparison[0, 0, 0, 0, 0]
            actual_torch = torch_norm[0, 0, 0, 0, 0]
        print(f"Manual verification - Expected: {expected_value:.6f}, Numpy: {actual_numpy:.6f}, Torch: {actual_torch:.6f}")
        
        # Test inverse normalization
        start_time = time.time()
        # We already have numpy_norm in the format expected by inverse_norm
        numpy_denorm = numpy_normalizer.inverse_norm(numpy_norm)
        numpy_time_inv = time.time() - start_time
        
        # Convert the denormalized result back to torch format for comparison
        if len(shape) == 4:  # (T, H, W, C) -> (C, T, H, W)
            numpy_denorm_for_comparison = numpy_denorm.transpose(3, 0, 1, 2)
        else:  # (B, T, H, W, C) -> (B, C, T, H, W)
            numpy_denorm_for_comparison = numpy_denorm.transpose(0, 4, 1, 2, 3)
        
        start_time = time.time()
        with torch.no_grad():
            torch_denorm = torch_normalizer.inverse_norm(torch.from_numpy(torch_norm).float()).numpy()
        torch_time_inv = time.time() - start_time
        
        max_diff_inverse = np.max(np.abs(numpy_denorm_for_comparison - torch_denorm))
        print(f"Inverse normalization max difference: {max_diff_inverse:.8f}")
        print(f"Numpy inverse time: {numpy_time_inv:.6f}s, Torch inverse time: {torch_time_inv:.6f}s")
        
        # Check if original data is recovered after normalization and denormalization
        original_recovery_numpy = np.max(np.abs(numpy_data - numpy_denorm_for_comparison))
        original_recovery_torch = np.max(np.abs(numpy_data - torch_denorm))
        print(f"Original data recovery (numpy): {original_recovery_numpy:.8f}")
        print(f"Original data recovery (torch): {original_recovery_torch:.8f}")
        
        # Assert tests pass with small tolerance
        tolerance = 1e-5
        assert max_diff < tolerance, f"Normalizers produce different results: {max_diff} > {tolerance}"
        assert max_diff_inverse < tolerance, f"Inverse normalizers produce different results: {max_diff_inverse} > {tolerance}"
        assert original_recovery_numpy < tolerance, f"Numpy original recovery failed: {original_recovery_numpy} > {tolerance}"
        assert original_recovery_torch < tolerance, f"Torch original recovery failed: {original_recovery_torch} > {tolerance}"

def test_hdf5_normalization(stats_path, norm_type='standard'):
    """Test HDF5 normalization functionality"""
    print(f"\n=== Testing {norm_type} HDF5 normalization ===")
    
    # Initialize normalizers
    numpy_normalizer = NormalizationApplier(stats_path, normalization_type=norm_type)
    torch_normalizer = TorchNormalizationApplier(stats_path, normalization_type=norm_type)
    
    # Get number of channels from stats
    n_channels = numpy_normalizer.n_channels
    
    # Create temporary HDF5 file
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create sample HDF5 file
        input_path = Path(temp_dir) / "test_input.h5"
        numpy_output_path = Path(temp_dir) / "numpy_output.h5"
        
        # Generate random data - using (batch, time, height, width, channels) for HDF5
        numpy_data, _ = create_sample_data((10, n_channels, 5, 8, 8))
        
        # Convert to HDF5 format (B, T, H, W, C)
        numpy_data_hdf5 = numpy_data.transpose(0, 2, 3, 4, 1)
        
        # Create HDF5 file
        with h5py.File(input_path, 'w') as f:
            f.create_dataset('data', data=numpy_data_hdf5)
        
        # Use numpy normalizer to normalize HDF5
        start_time = time.time()
        numpy_normalizer.normalize_hdf5(
            input_path=input_path,
            output_path=numpy_output_path,
            key='data',
            batch_size=5
        )
        hdf5_time = time.time() - start_time
        print(f"HDF5 normalization time: {hdf5_time:.6f}s")
        
        # Load normalized data
        with h5py.File(numpy_output_path, 'r') as f:
            numpy_normalized = f['data'][()]
        
        # Compare with torch normalization
        # For torch, convert from (B, T, H, W, C) to (B, C, T, H, W)
        numpy_data_torch_format = numpy_data
        
        start_time = time.time()
        with torch.no_grad():
            torch_normalized = torch_normalizer(torch.from_numpy(numpy_data_torch_format).float()).numpy()
        torch_time = time.time() - start_time
        print(f"Torch direct normalization time: {torch_time:.6f}s")
        
        # Convert torch result back to HDF5 format (B, C, T, H, W) -> (B, T, H, W, C)
        torch_normalized_hdf5_format = torch_normalized.transpose(0, 2, 3, 4, 1)
        
        max_diff = np.max(np.abs(numpy_normalized - torch_normalized_hdf5_format))
        print(f"HDF5 normalization max difference: {max_diff:.8f}")
        
        # Assert tests pass with small tolerance
        tolerance = 1e-5
        assert max_diff < tolerance, f"HDF5 normalization differs from direct torch normalization: {max_diff} > {tolerance}"

def visualize_normalization(stats_path):
    """Visualize the effects of normalization on sample data"""
    print("\n=== Visualizing normalization effects ===")
    
    # Create normalizers for both types
    std_normalizer = TorchNormalizationApplier(stats_path, normalization_type='standard')
    minmax_normalizer = TorchNormalizationApplier(stats_path, normalization_type='minmax')
    
    # Get number of channels from stats
    with open(stats_path, 'r') as f:
        stats = json.load(f)
        n_channels = len(stats)
    
    # Create a simple increasing pattern for easy visualization
    x = np.linspace(-2, 2, 100).astype(np.float32)
    data = np.zeros((n_channels, 1, 1, 100), dtype=np.float32)
    
    for c in range(n_channels):
        data[c, 0, 0, :] = x + 0.5 * c  # Offset each channel for visibility
    
    # Convert to torch tensor
    torch_data = torch.from_numpy(data).float()
    
    # Apply normalizations
    with torch.no_grad():
        std_normalized = std_normalizer(torch_data).numpy()
        minmax_normalized = minmax_normalizer(torch_data).numpy()
    
    # Plot the results
    plt.figure(figsize=(15, 10))
    
    # Original data
    plt.subplot(3, 1, 1)
    for c in range(n_channels):
        plt.plot(data[c, 0, 0, :], label=f'Channel {c}')
    plt.title('Original Data')
    plt.legend()
    plt.grid(True)
    
    # Standard normalization
    plt.subplot(3, 1, 2)
    for c in range(n_channels):
        plt.plot(std_normalized[c, 0, 0, :], label=f'Channel {c}')
    plt.title('Standard Normalization (Z-score)')
    plt.legend()
    plt.grid(True)
    
    # MinMax normalization
    plt.subplot(3, 1, 3)
    for c in range(n_channels):
        plt.plot(minmax_normalized[c, 0, 0, :], label=f'Channel {c}')
    plt.title('MinMax Normalization (Range [-1, 1])')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('normalization_visualization.png')
    print("Saved visualization to normalization_visualization.png")

def main():
    parser = argparse.ArgumentParser(description="Test numpy and torch normalizers for equality")
    parser.add_argument("--stats_path", type=Path, 
                        help="Path to normalization stats JSON file (optional, will create temporary one if not provided)")
    
    args = parser.parse_args()
    
    # Create a temporary directory for test files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Use provided stats or create temporary one
        if args.stats_path:
            stats_path = args.stats_path
        else:
            stats_path = create_test_stats_file(temp_dir)
            print(f"Created temporary stats file: {stats_path}")
        
        # Test standard normalization
        test_normalization_equality(stats_path, norm_type='standard')
        
        # Test minmax normalization
        test_normalization_equality(stats_path, norm_type='minmax')
        
        # Test HDF5 functionality
        test_hdf5_normalization(stats_path, norm_type='standard')
        test_hdf5_normalization(stats_path, norm_type='minmax')
        
        # Visualize the normalization effects
        visualize_normalization(stats_path)
        
        print("\nAll tests passed! Both normalizers produce equivalent results.")

if __name__ == "__main__":
    main()


"""

python -m well_utils.data_processing.normalization.testing

"""