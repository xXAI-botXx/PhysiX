import h5py
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
from typing import List, Dict
import argparse


class StatsCollector:
    """Collects running statistics for multiple data channels"""
    def __init__(self, n_channels: int):
        self.n_channels = n_channels
        self.min = np.full(n_channels, np.inf)
        self.max = np.full(n_channels, -np.inf)
        self.sum = np.zeros(n_channels)
        self.mean = np.zeros(n_channels)
        self.m2 = np.zeros(n_channels)  # For variance calculation
        self.count = 0

    def update(self, batch: np.ndarray) -> None:
        """Update statistics with a batch of data"""
        batch = batch.reshape(-1, self.n_channels)
        batch_size = batch.shape[0]
        
        # Update range statistics
        self.min = np.minimum(self.min, batch.min(axis=0))
        self.max = np.maximum(self.max, batch.max(axis=0))
        
        # Welford's algorithm for online variance
        delta = batch - self.mean
        self.mean += delta.sum(axis=0) / (self.count + batch_size)
        delta2 = batch - self.mean
        self.m2 += (delta * delta2).sum(axis=0)
        self.sum += batch.sum(axis=0)
        self.count += batch_size

    def get_stats(self) -> List[Dict]:
        """Return formatted statistics for each channel"""
        return [{
            'min': float(self.min[i]),
            'max': float(self.max[i]),
            'mean': float(self.mean[i]),
            'std': float(np.sqrt(self.m2[i] / (self.count - 1)) if self.count > 1 else 0),
            'count': self.count
        } for i in range(self.n_channels)]

def compute_and_save_stats(
    input_dir: Path,
    output_path: Path,
    key: str = 'data',
    batch_size: int = 1000,
    limit: int = None
) -> None:
    """
    Compute statistics from HDF5 files and save to JSON
    Args:
        input_dir: Directory containing HDF5 files
        output_path: Path to save statistics JSON file
        key: Dataset key in HDF5 files
        batch_size: Processing batch size
        limit: Maximum number of files to process
    """
    files = _discover_hdf5_files(input_dir, limit)
    collector = None
    
    for fpath in tqdm(files, desc="Analyzing files"):
        with h5py.File(fpath, 'r') as f:
            if key not in f:
                continue
                
            data = f[key]
            if collector is None:
                collector = StatsCollector(data.shape[-1])
            
            for i in range(0, len(data), batch_size):
                collector.update(data[i:i+batch_size][()])
    
    if collector is None:
        raise ValueError("No valid HDF5 files found with specified key")
    
    stats = collector.get_stats()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2)

def _discover_hdf5_files(input_dir: Path, limit: int = None) -> List[Path]:
    """Discover HDF5 files in directory structure"""
    files = list(input_dir.rglob("*.hdf5"))
    print(f"Found {len(files)} HDF5 files in {input_dir}")
    return files[:limit] if limit else files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate dataset statistics")
    parser.add_argument("--input_dir", required=True, type=Path, 
                      help="Directory containing HDF5 files")
    parser.add_argument("--output_path", required=True, type=Path,
                      help="Path to save statistics JSON file")
    parser.add_argument("--key", default="data", type=str,
                      help="HDF5 dataset key")
    parser.add_argument("--batch_size", default=1000, type=int,
                      help="Processing batch size")
    parser.add_argument("--limit", default=None, type=int,
                      help="Limit number of files processed")
    
    args = parser.parse_args()
    compute_and_save_stats(
        input_dir=args.input_dir,
        output_path=args.output_path,
        key=args.key,
        batch_size=args.batch_size,
        limit=args.limit
    )
    print(f"Statistics saved to {args.output_path}")