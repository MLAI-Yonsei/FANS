import os
import numpy as np
from pathlib import Path
import glob
import shutil

# Set global random seed
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

def subsample_data(input_dir, output_dir, n_samples=1000):
    """
    Subsample data files and save them to a new directory.
    
    Parameters:
    input_dir (str): Path to the directory containing original data
    output_dir (str): Path to the directory to save subsampled data
    n_samples (int): Number of samples to extract (default: 1000)
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Iterate through all node directories
    node_dirs = [d for d in os.listdir(input_dir) if d.startswith('nodes_')]
    
    for node_dir in node_dirs:
        node_path = os.path.join(input_dir, node_dir)
        if not os.path.isdir(node_path):
            continue
            
        print(f"Processing: {node_dir}")
        
        # Graph types within each node directory (ER, SF, etc.)
        graph_types = [d for d in os.listdir(node_path) if os.path.isdir(os.path.join(node_path, d))]
        
        for graph_type in graph_types:
            graph_path = os.path.join(node_path, graph_type)
            print(f"  Processing: {graph_type}")
            
            # Create output directory structure
            output_graph_path = os.path.join(output_dir, node_dir, graph_type)
            os.makedirs(output_graph_path, exist_ok=True)
            
            # Get all files
            all_files = [f for f in os.listdir(graph_path) if os.path.isfile(os.path.join(graph_path, f))]
            
            # Classify and process files
            for filename in all_files:
                file_path = os.path.join(graph_path, filename)
                
                # Subsample data_env1_*.npy and data_env2_*.npy files
                if filename.startswith('data_env1_') and filename.endswith('.npy'):
                    process_file(file_path, output_graph_path, n_samples)
                elif filename.startswith('data_env2_') and filename.endswith('.npy'):
                    process_file(file_path, output_graph_path, n_samples)
                else:
                    # Copy remaining files as is
                    copy_file(file_path, output_graph_path)

def process_file(file_path, output_dir, n_samples):
    """
    Process and subsample individual files.
    
    Parameters:
    file_path (str): Path to the file to process
    output_dir (str): Output directory
    n_samples (int): Number of samples to extract
    """
    try:
        # Extract filename
        filename = os.path.basename(file_path)
        
        # Load data
        data = np.load(file_path)
        print(f"    Subsampling: {filename} (original size: {data.shape})")
        
        # Use entire data if number of samples is less than requested
        if len(data) <= n_samples:
            subsampled_data = data
            print(f"      Warning: Number of samples in {filename} ({len(data)}) is less than requested ({n_samples}). Using entire dataset.")
        else:
            # Randomly select n_samples
            indices = np.random.choice(len(data), n_samples, replace=False)
            subsampled_data = data[indices]
        
        # Save result
        output_path = os.path.join(output_dir, filename)
        np.save(output_path, subsampled_data)
        print(f"      Saved: {output_path} (new size: {subsampled_data.shape})")
        
    except Exception as e:
        print(f"      Error occurred in {filename}: {str(e)}")

def copy_file(file_path, output_dir):
    """
    Copy file as is.
    
    Parameters:
    file_path (str): Path to the file to copy
    output_dir (str): Output directory
    """
    # Extract filename
    filename = os.path.basename(file_path)
    
    # Set output path
    output_path = os.path.join(output_dir, filename)
    
    # Copy file
    shutil.copy2(file_path, output_path)
    print(f"    Copied: {filename}")

def main():
    """
    Main execution function
    """
    # Reset seed (explicit confirmation)
    np.random.seed(RANDOM_SEED)
    
    # Set paths
    input_dir = 'data/'
    output_dir = 'data_small'
    n_samples = 1000
    
    print(f"Starting data processing...")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Number of samples to subsample: {n_samples}")
    print(f"Random seed: {RANDOM_SEED}")
    print("Processing method:")
    print("  - data_env1_*.npy, data_env2_*.npy: subsampling")
    print("  - Other files: copy as is")
    print("-" * 50)
    
    # Execute subsampling
    subsample_data(input_dir, output_dir, n_samples)
    
    print("-" * 50)
    print("Data processing complete!")

if __name__ == "__main__":
    main()