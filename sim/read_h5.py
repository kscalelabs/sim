import h5py
import os
import argparse
import numpy as np

def print_h5_contents(file_path):
    """Print the contents and structure of an HDF5 file."""
    with h5py.File(file_path, 'r') as f:
        print(f"\nReading HDF5 file: {file_path}")
        print("\nFile structure:")
        
        def print_dataset_info(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"\nDataset: {name}")
                print(f"  Shape: {obj.shape}")
                print(f"  Type: {obj.dtype}")
                
                # Print a small sample of data
                if len(obj.shape) > 0:
                    sample_size = min(3, obj.shape[0])  # Show first 3 entries or less
                    print(f"  First {sample_size} entries:")
                    print(f"    {obj[:sample_size]}")
        
        # Recursively visit all groups and datasets
        f.visititems(print_dataset_info)


        

def main():
    parser = argparse.ArgumentParser(description='Read and display contents of HDF5 file')
    parser.add_argument('--embodiment', type=str, required=True, help='Embodiment directory name')
    args = parser.parse_args()
    
    # Check if directory exists
    if not os.path.exists(args.embodiment):
        print(f"Error: Directory '{args.embodiment}' not found")
        return
    
    # List all h5 files in the directory
    h5_files = [f for f in os.listdir(args.embodiment) if f.endswith('.h5')]
    
    if not h5_files:
        print(f"No HDF5 files found in {args.embodiment}/")
        return
    
    # Print contents of each file
    for h5_file in h5_files:
        file_path = os.path.join(args.embodiment, h5_file)
        print_h5_contents(file_path)

if __name__ == "__main__":
    main() 