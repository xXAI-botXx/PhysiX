import os
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
from config import RAW_DATA_PATH, CLEANED_DATA_PATH

from specialized_processors import processor_mapping

def process_well_dataset(input_dir, output_dir, dataset_name):
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(input_dir)
    
    total_files = sum([len(files) for _, _, files in os.walk(input_dir)])
    
    with tqdm(total=total_files, desc="Processing files") as pbar:
        for subdir, _, files in os.walk(input_dir):
            subfolder_name = Path(subdir).relative_to(input_dir)
            output_subfolder = output_dir / subfolder_name
            output_subfolder.mkdir(parents=True, exist_ok=True)
            
            for file in files:
                if not "input" in str(file):
                    continue
                input_file_path = Path(subdir) / file
                try:
                    processor_mapping[dataset_name](input_file_path, output_subfolder, chunked=False)
                    os.remove(input_file_path)
                    pbar.update(1)
                except Exception as e:
                    print(f"Error processing {input_file_path}: {e}")
                    raise e
                    pbar.update(1)

def clean_simulation_data(dataset_name, raw_data_path, cleaned_data_path):
    process_well_dataset(raw_data_path / dataset_name / 'data', cleaned_data_path, dataset_name)

def main():
    parser = argparse.ArgumentParser(description="Process and clean simulation data.")
    parser.add_argument("dataset_name", type=str, help="Name of the dataset to process.")
    parser.add_argument("--raw_data_path", type=str, default=RAW_DATA_PATH, 
                        help="Path to the raw data directory.")
    parser.add_argument("--cleaned_data_path", type=str, default=CLEANED_DATA_PATH, 
                        help="Path to the cleaned data directory.")
    
    args = parser.parse_args()
    
    clean_simulation_data(args.dataset_name, Path(args.raw_data_path), Path(args.cleaned_data_path))

if __name__ == "__main__":
    main()