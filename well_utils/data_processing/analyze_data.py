import h5py
import argparse

def print_shapes(group, current_path):
    for key in group.keys():
        item = group[key]
        if isinstance(item, h5py.Group):
            print_shapes(item, f"{current_path}{key}/")
        elif isinstance(item, h5py.Dataset):
            print(f"{current_path}{key} {item.shape}", "Min:", item[()].min(), "Max:", item[()].max(), "Mean:", item[()].mean(), "Magnitude:", abs(item[()]).mean())

def main():
    parser = argparse.ArgumentParser(description='Print shapes of arrays in an HDF5 file.')
    parser.add_argument('file', type=str, help='Path to the HDF5 file')
    args = parser.parse_args()

    with h5py.File(args.file, 'r') as f:
        print_shapes(f, '')

if __name__ == '__main__':
    main()