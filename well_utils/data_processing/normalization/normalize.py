import argparse
from pathlib import Path
import h5py
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
from typing import Union

from well_utils.data_processing.normalization.calculate_stats import compute_and_save_stats, _discover_hdf5_files

class NormalizationApplier:
    """Applies normalization and denormalization using precomputed statistics"""
    def __init__(self, stats_path: Path, normalization_type: str = 'standard'):
        with open(stats_path, 'r') as f:
            self.stats = json.load(f)
        
        self.n_channels = len(self.stats)
        self.normalization_type = normalization_type

        if self.normalization_type == 'minmax':
            self.ranges = [(c['min'], c['max']) for c in self.stats]
            self.scale = np.array([2 / (max_ - min_ + 1e-7) for min_, max_ in self.ranges])
            self.offset = np.array([-1 - (2 * min_)/(max_ - min_ + 1e-7) for min_, max_ in self.ranges])
        elif self.normalization_type == 'standard':
            self.means = np.array([c['mean'] for c in self.stats])
            self.stds = np.array([c['std'] for c in self.stats])
            self.scale = np.array([1 / (std + 1e-7) for std in self.stds])
            self.offset = np.array([-mean / (std + 1e-7) for mean, std in zip(self.means, self.stds)])
        else:
            raise ValueError(f"Unsupported normalization type: {normalization_type}")

    def normalize(self, arr: np.ndarray) -> np.ndarray:
        """Normalize a numpy array using stored statistics"""
        self._validate_array_shape(arr)
        return arr * self.scale + self.offset

    def inverse_norm(self, normalized_arr: np.ndarray) -> np.ndarray:
        """inverse_norm a numpy array using stored statistics"""
        self._validate_array_shape(normalized_arr)
        return (normalized_arr - self.offset) / self.scale

    def normalize_hdf5(
        self,
        input_path: Path,
        output_path: Path,
        key: str = 'data',
        batch_size: int = 1000,
        delete_input: bool = False
    ) -> None:
        """Normalize an HDF5 file and save result"""
        self._process_hdf5(
            input_path=input_path,
            output_path=output_path,
            key=key,
            batch_size=batch_size,
            process_fn=self.normalize,
            desc="Normalizing",
            delete_input=delete_input
        )

    def inverse_norm_hdf5(
        self,
        input_path: Path,
        output_path: Path,
        key: str = 'data',
        batch_size: int = 1000,
        delete_input: bool = False
    ) -> None:
        """inverse_norm an HDF5 file and save result"""
        self._process_hdf5(
            input_path=input_path,
            output_path=output_path,
            key=key,
            batch_size=batch_size,
            process_fn=self.inverse_norm,
            desc="Denormalizing",
            delete_input=delete_input
        )

    def _validate_array_shape(self, arr: np.ndarray) -> None:
        """Validate array shape matches expected number of channels"""
        if arr.shape[-1] != self.n_channels:
            raise ValueError(f"Array has {arr.shape[-1]} channels, expected {self.n_channels}")

    def _process_hdf5(
        self,
        input_path: Path,
        output_path: Path,
        key: str,
        batch_size: int,
        process_fn: callable,
        desc: str,
        delete_input: bool = False
    ) -> None:
        """Generic HDF5 processing function"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with h5py.File(input_path, 'r') as f_in, h5py.File(output_path, 'w') as f_out:
            if key not in f_in:
                raise KeyError(f"Key '{key}' not found in input file")
            
            data = f_in[key]
            dset_out = f_out.create_dataset(key, data.shape, dtype=np.float32, chunks=True)
            
            for i in tqdm(range(0, len(data), batch_size), desc=desc):
                batch = data[i:i+batch_size][()]
                processed = process_fn(batch)
                dset_out[i:i+batch_size] = processed
        
        if delete_input:
            input_path.unlink()  # Delete the input file after successful processing

def load_normalizer(stats_path: Union[str, Path], normalization_type: str = 'minmax') -> NormalizationApplier:
    """Helper function to create normalizer from stats file"""
    return NormalizationApplier(Path(stats_path), normalization_type)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Normalize HDF5 datasets")
    parser.add_argument("--input_dir", required=True, type=Path,
                      help="Directory containing HDF5 files to normalize")
    parser.add_argument("--output_dir", required=True, type=Path,
                      help="Directory to save normalized files")
    parser.add_argument("--stats_path", type=Path,
                      help="Path to existing statistics file (optional)")
    parser.add_argument("--key", default="data", type=str,
                      help="HDF5 dataset key")
    parser.add_argument("--batch_size", default=1000, type=int,
                      help="Normalization batch size")
    parser.add_argument("--limit", default=None, type=int,
                      help="Limit number of files processed")
    parser.add_argument("--normalization_type", choices=['minmax', 'standard'], default='minmax',
                      help="Type of normalization to apply (minmax or standard)")
    parser.add_argument("--delete", action="store_true", 
                      help="Delete input files after successful normalization")
    
    args = parser.parse_args()
    
    # Calculate stats if not provided
    if not args.stats_path:
        stats_path = args.output_dir / "normalization_stats.json"
        print("No stats file provided, calculating from input data...")
        from well_utils.data_processing.normalization.calculate_stats import compute_and_save_stats
        compute_and_save_stats(
            input_dir=args.input_dir,
            output_path=stats_path,
            key=args.key,
            batch_size=args.batch_size,
            limit=args.limit
        )
        args.stats_path = stats_path
    
    # Normalize files
    normalizer = load_normalizer(args.stats_path, args.normalization_type)
    
    files = _discover_hdf5_files(args.input_dir, args.limit)
    print(f"Normalizing {len(files)} files...")
    
    for fpath in tqdm(files, desc="Processing"):
        rel_path = fpath.relative_to(args.input_dir)
        output_path = args.output_dir / rel_path
        normalizer.normalize_hdf5(
            input_path=fpath,
            output_path=output_path,
            key=args.key,
            batch_size=args.batch_size,
            delete_input=args.delete
        )
    
    print(f"Normalized files saved to {args.output_dir}")