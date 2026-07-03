import os
import json
import numpy as np
import pandas as pd
import yaml
import argparse
import torch
import gc
import time
from utils import dict_to_namespace, parse_dataset_indices, create_results_dir, make_json_serializable, setup_gpu
from baselines import iscan
from baselines.linearccp import run_linearccp_analysis
from baselines.splitkci.dependence_measures import KCIMeasure
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def run_iscan_model(data_env1, data_env2, adj, metadata, device):
    print("Running ISCAN model...")
    shift_detection_start = time.time()
    shifted_nodes_est, order, ratio_dict = iscan.est_node_shifts(data_env1, data_env2, elbow=True)
    shift_detection_time = time.time() - shift_detection_start

    # Store ISCAN results (unified naming)
    results = {
        "shifted_nodes": shifted_nodes_est,
        "order": order,
        "ratio_dict": ratio_dict,
        "shift_detection_time": shift_detection_time
    }
    
    # Get true shifted nodes from metadata (optional)
    true_shifted = set(metadata.get("shifted_nodes", []))
    detected = set([int(n) for n in shifted_nodes_est])
    
    functional_shift_start = time.time()
    # Test functional shifts for ALL true shifted nodes regardless of detection status
    if true_shifted:
        print(f"Testing functional shifts for ALL true shifted nodes, regardless of detection status")
        
        # Variables to store results
        functional_shift_results = {}
        true_node_results = {}
        
        # Record results for each true shifted node
        for true_node in true_shifted:
            is_detected = true_node in detected
            has_functional_shift = False

            # Find all parents of the shifted node from DAG
            parents = np.where(adj[:, true_node] == 1)[0].tolist()
                
            if parents:  # Only test if node has parents
                print(f"Testing functional shifts for true shifted node {true_node} with parents: {parents}")
                    
                # Find functionally shifted edges focused on this shifted node
                shift_detected, f_shifted_edges, test_details = iscan.find_functionally_shifted_edges_with_dag(
                    y_node=true_node,
                    parents_y=parents,
                    data_env1=data_env1,
                    data_env2=data_env2
                )

                has_functional_shift = shift_detected
                
                # Store functional shift results
                functional_shift_results[str(true_node)] = {
                    "shifted_edges": f_shifted_edges,
                    "test_details": test_details,
                    "has_functional_shift": has_functional_shift
                }
            
            # Store results for true shifted node
            true_node_results[true_node] = {
                "is_detected_as_shifted": is_detected,
                "has_functional_shift_detected": has_functional_shift if parents else "No parents to test"
            }
        functional_shift_time = time.time() - functional_shift_start
        # Add results to output
        results["true_shifted_node_summary"] = true_node_results
        results["functional_shift_results"] = functional_shift_results
        results["functional_shift_time"] = functional_shift_time

        # Aggregate results
        all_shifted_edges = []
        for node_results in functional_shift_results.values():
            if node_results.get("shifted_edges"):
                all_shifted_edges.extend(node_results["shifted_edges"])
        
        results["all_shifted_edges"] = all_shifted_edges
        
        # Create unified estimated_shift_types (function or noise)
        estimated_shift_types = {}
        for true_node in true_shifted:
            node_str = str(true_node)
            if node_str in functional_shift_results:
                if functional_shift_results[node_str].get("has_functional_shift", False):
                    estimated_shift_types[node_str] = "function"
                else:
                    estimated_shift_types[node_str] = "noise"
        results["estimated_shift_types"] = estimated_shift_types
    else:
        print("No ground truth shifted nodes available - skipping functional shift tests")
            
    return results

def run_splitkci_model(data_env1, data_env2, adj, metadata, alpha=0.05, train_test_split=0.001):
    """
    Run SplitKCI (Split Kernel Conditional Independence) test
    
    Tests Y ⟂ Z | X where:
    - Y: child node
    - Z: environment (0 or 1)
    - X: parent nodes
    
    If p-value < alpha, the conditional distribution Y|X differs across environments
    → shift detected
    
    Args:
        data_env1: Environment 1 data
        data_env2: Environment 2 data
        adj: Adjacency matrix
        metadata: Dataset metadata
        alpha: Significance level (default: 0.05)
        train_test_split: Train/test split ratio for KCI (default: 0.001)
    
    Returns:
        dict: SplitKCI analysis results
    """
    print("Running SplitKCI model...")
    
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Get parent-child relationships
    parent_child_pairs = []
    for child in range(adj.shape[0]):
        parents = np.where(adj[:, child] == 1)[0]
        if len(parents) > 0:
            parent_child_pairs.append((child, parents))
    
    print(f"Found {len(parent_child_pairs)} parent-child pairs")
    
    # Shift detection
    detected_shifts = []
    shift_details = {}
    detection_start_time = time.time()
    
    for child, parents in parent_child_pairs:
        print(f"Testing node {child} with {len(parents)} parents...")
        
        # Prepare data
        # X = parents, Y = child, Z = environment (0/1)
        parent_data_env1 = data_env1[:, parents]
        parent_data_env2 = data_env2[:, parents]
        child_data_env1 = data_env1[:, child]
        child_data_env2 = data_env2[:, child]
        
        # Combine samples (2n, ·)
        parent_data = np.vstack([parent_data_env1, parent_data_env2])
        child_data = np.hstack([child_data_env1, child_data_env2])
        env_labels = np.array([0] * len(child_data_env1) + [1] * len(child_data_env2))
        
        # Adjust dimensions
        if env_labels.ndim == 1:
            env_labels = env_labels[:, None]
        if child_data.ndim == 1:
            child_data = child_data[:, None]
        
        # Convert to tensors
        parent_data_t = torch.tensor(parent_data, dtype=torch.float32)
        child_data_t = torch.tensor(child_data, dtype=torch.float32)
        env_labels_t = torch.tensor(env_labels, dtype=torch.float32)
        
        # Kernel parameters (using variance)
        sigma_x = float(parent_data_t.var().item() + 1e-8)
        sigma_y = float(child_data_t.var().item() + 1e-8)
        sigma_z = float(env_labels_t.var().item() + 1e-8)
        
        kernel_a_args = {"sigma2": sigma_y}       # a = Y (child)
        kernel_b_args = {"sigma2": sigma_z}       # b = Z (environment)
        kernel_ca_args = {"sigma2": sigma_x}      # c = X (parents) for Y regression
        kernel_cb_args = {"sigma2": sigma_x}      # c = X (parents) for Z regression
        
        try:
            kci = KCIMeasure(
                kernel_a="gaussian",
                kernel_ca="gaussian",
                kernel_b="gaussian",
                kernel_cb="gaussian",
                kernel_ca_args=kernel_ca_args,
                kernel_a_args=kernel_a_args,
                kernel_cb_args=kernel_cb_args,
                kernel_b_args=kernel_b_args,
                biased=True
            )
            
            # Test Y ⟂ Z | X
            p_value = kci.test(child_data_t, env_labels_t, parent_data_t, train_test_split=train_test_split)
            
            shift_details[str(child)] = {
                "p_value": float(p_value),
                "num_parents": len(parents),
                "parents": parents.tolist(),
                "shift_detected": p_value < alpha
            }
            
            if p_value < alpha:
                detected_shifts.append(child)
                print(f"  → Shift detected! p-value: {p_value:.4f}")
            else:
                print(f"  → No shift. p-value: {p_value:.4f}")
                
        except Exception as e:
            print(f"  → Error testing node {child}: {str(e)}")
            shift_details[str(child)] = {
                "error": str(e),
                "num_parents": len(parents),
                "parents": parents.tolist()
            }
    
    detection_end_time = time.time()
    detection_time = detection_end_time - detection_start_time
    
    results = {
        "shifted_nodes": detected_shifts,  # unified naming
        "shift_details": shift_details,
        "num_parent_child_pairs": len(parent_child_pairs),
        "detection_time_seconds": detection_time,
        "alpha": alpha,
        "train_test_split": train_test_split
    }
    
    print(f"SplitKCI completed in {detection_time:.2f} seconds")
    print(f"Detected shifts: {detected_shifts}")
    
    return results

# ----- Data Loading Functions -----

def load_morpho_mnist_data():
    """Load Morpho-MNIST dataset"""
    print("\n===== Loading Morpho-MNIST dataset =====")
    
    data_dir = "/home/statduck/fans/data/morpho_mnist"
    
    data_env1 = np.load(f"{data_dir}/data_env1_test.npy")
    data_env2 = np.load(f"{data_dir}/data_env2.npy")
    adj = np.load(f"{data_dir}/adj_morpho_mnist.npy")
    
    print(f"Loaded data - Env1 shape: {data_env1.shape}, Env2 shape: {data_env2.shape}, Adj shape: {adj.shape}")
    
    metadata = {
        "dataset_type": "morpho_mnist",
        "data_shapes": {
            "env1": list(data_env1.shape),
            "env2": list(data_env2.shape),
            "adj": list(adj.shape)
        }
    }
    
    # Return as list with single dataset for consistent interface
    return [{
        "data_env1": data_env1,
        "data_env2": data_env2,
        "adj": adj,
        "metadata": metadata,
        "dataset_name": "morpho_mnist",
        "config_name": "main",
        "dataset_idx": None,
        "node_count": adj.shape[0]
    }]

def load_synthetic_datasets(node_counts, config_type, dataset_indices_filter, data_dir):
    """Load synthetic datasets based on specified parameters"""
    print("\n===== Loading Synthetic datasets =====")
    print(f"Node counts: {node_counts}")
    print(f"Config type: {config_type}")
    if dataset_indices_filter:
        print(f"Dataset indices filter: {dataset_indices_filter}")
    
    datasets = []
    
    # Filter configurations
    if config_type == 'all':
        configurations = ["ER", "SF"]
    else:
        configurations = [config_type]
    
    print(f"Configurations: {configurations}")
    
    for node_count in node_counts:
        for config_name in configurations:
            data_path = f"{data_dir}/nodes_{node_count}/{config_name}"
            
            if not os.path.exists(data_path):
                print(f"Data directory {data_path} not found, skipping...")
                continue
            
            # Find all dataset indices
            dataset_indices = []
            for file in os.listdir(data_path):
                if file.startswith("metadata_") and file.endswith(".json"):
                    idx = int(file.replace("metadata_", "").replace(".json", ""))
                    dataset_indices.append(idx)
            
            dataset_indices.sort()
            
            # Filter dataset indices if specified
            selected_indices = parse_dataset_indices(dataset_indices_filter, dataset_indices)
            
            print(f"Loading {len(selected_indices)} datasets from {data_path}")
            
            for dataset_idx in selected_indices:
                # Load data
                data_env1 = np.load(f"{data_path}/data_env1_{dataset_idx}.npy")
                data_env2 = np.load(f"{data_path}/data_env2_{dataset_idx}.npy")
                adj = np.load(f"{data_path}/adj_{dataset_idx}.npy")
                
                with open(f"{data_path}/metadata_{dataset_idx}.json", 'r') as f:
                    metadata = json.load(f)
                
                datasets.append({
                    "data_env1": data_env1,
                    "data_env2": data_env2,
                    "adj": adj,
                    "metadata": metadata,
                    "dataset_name": f"nodes_{node_count}",
                    "config_name": config_name,
                    "dataset_idx": dataset_idx,
                    "node_count": node_count
                })
    
    print(f"Total datasets loaded: {len(datasets)}")
    return datasets

# ----- Model Execution Function -----

def run_model_on_data(model_name, data_env1, data_env2, adj, metadata, args, device, output_dir=None):
    """Execute specified model on given data - unified interface for all models"""
    if model_name == 'iscan':
        return run_iscan_model(data_env1, data_env2, adj, metadata, device)
    
    elif model_name == 'linearccp':
        return run_linearccp_analysis(data_env1, data_env2, adj, metadata)
    
    elif model_name == 'splitkci':
        return run_splitkci_model(data_env1, data_env2, adj, metadata)
    
    else:
        raise ValueError(f"Unknown model: {model_name}")

# ----- Dataset Processing Function -----

def process_single_dataset(dataset_info, args, device, main_results_dir):
    """Process a single dataset and save results"""
    # Extract dataset information
    data_env1 = dataset_info["data_env1"]
    data_env2 = dataset_info["data_env2"]
    adj = dataset_info["adj"]
    metadata = dataset_info["metadata"]
    dataset_name = dataset_info["dataset_name"]
    config_name = dataset_info["config_name"]
    dataset_idx = dataset_info["dataset_idx"]
    node_count = dataset_info["node_count"]
    
    # Print dataset info
    if dataset_idx is not None:
        print(f"\nProcessing dataset {dataset_idx} - {dataset_name}/{config_name}")
        print(f"Nodes: {node_count}, Shifted nodes: {metadata.get('shifted_nodes', [])}")
    else:
        print(f"\nProcessing {dataset_name}")
    
    # Determine save path and filename
    if args.exp_type == 'morpho_mnist':
        result_filename = f"{args.model}_{dataset_name}.json"
        save_path = f"{main_results_dir}/{result_filename}"
        os.makedirs(main_results_dir, exist_ok=True)
    else:  # synthetic
        result_filename = f"{args.model}_nodes{node_count}_{config_name}_{dataset_idx}.json"
        config_results_dir = f"{main_results_dir}/nodes_{node_count}/{config_name}/{args.model}"
        os.makedirs(config_results_dir, exist_ok=True)
        save_path = f"{config_results_dir}/{result_filename}"
    
    # Run model
    print(f"Running {args.model} model...")
    model_results = run_model_on_data(
        args.model, data_env1, data_env2, adj, metadata, args, device,
        output_dir=main_results_dir
    )
    success = True
    error_msg = None
    
    # Initialize results dictionary
    results = {
        "dataset_info": metadata,
        "model": args.model,
        "exp_type": args.exp_type,
        "config_name": config_name,
        "device": str(device),
        args.model: model_results,
        "success": success,
        "error": error_msg
    }
    
    # Save results as JSON
    with open(save_path, 'w') as f:
        json.dump(make_json_serializable(results), f, indent=2)
    
    print(f"✓ Results saved to {save_path}")
    
    return results

# ----- Main Function -----

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Run experiments with selected model')
    parser.add_argument('--model', type=str, choices=['iscan', 'linearccp', 'splitkci'], required=True,
                      help='Model to run: iscan, linearccp, or splitkci')
    parser.add_argument('--exp_type', type=str, choices=['synthetic', 'morpho_mnist'], default='synthetic',
                      help='Experiment type: synthetic or morpho_mnist (default: synthetic)')
    parser.add_argument('--nodes', type=int, nargs='+', default=[10, 20, 30, 40, 50], 
                      help='Node counts to process (default: 10 20 30 40 50)')
    parser.add_argument('--gpu', type=int, default=-1,
                      help='GPU ID to use (-1 for CPU, default: -1)')
    parser.add_argument('--output_dir', type=str, default=None,
                      help='Output directory for results (default: auto-generated with timestamp)')
    parser.add_argument('--config_type', type=str, choices=['ER','SF'], default='all',
                      help='Filter experiments by configuration type (default: all)')
    parser.add_argument('--dataset_indices', type=str, default=None,
                      help='Indices of datasets to process (e.g., "0-5" for datasets 0,1,2,3,4,5)')
    args = parser.parse_args()

    # Set up device (GPU/CPU)
    device = setup_gpu(args.gpu)

    # Print experiment configuration
    print(f"Running experiments with model: {args.model}")
    print(f"Experiment type: {args.exp_type}")
    print(f"Using device: {device}")

    # Create results directory
    main_results_dir = create_results_dir(args.output_dir, args.model)

    # Load datasets based on experiment type
    if args.exp_type == 'morpho_mnist':
        datasets = load_morpho_mnist_data()
    else:  # synthetic
        data_dir = "/mlainas/statduck/data_small"
        datasets = load_synthetic_datasets(
            node_counts=args.nodes,
            config_type=args.config_type,
            dataset_indices_filter=args.dataset_indices,
            data_dir=data_dir
        )
    
    # Process all datasets
    print(f"\n===== Processing {len(datasets)} dataset(s) =====")
    
    for dataset_info in datasets:
        process_single_dataset(dataset_info, args, device, main_results_dir)
    
    # Final summary
    print(f"\n===== All experiments completed =====")
    print(f"Results saved to {main_results_dir}/")
    print(f"Total datasets processed: {len(datasets)}")
    print(f"GPU usage: {device}")

if __name__ == "__main__":
    main()

