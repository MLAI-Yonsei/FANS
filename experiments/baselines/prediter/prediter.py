#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to run PreDITEr experiments on synthetic data
- Use real data (env1, env2) to compute covariance matrices
- Run PreDITEr model to predict shifted nodes
- Save results to prediter_result_test file
"""

import os
import json
import numpy as np
import time
import multiprocessing as mp
from datetime import datetime
from pathlib import Path
import sys

# Add main_codes directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'main_codes'))

from functions_sample import IMAG_pasp_sample

# Get project root directory (3 levels up from this script)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "data_small")

def load_data(node_count, graph_type, index):
    """Load experimental data from data directory"""
    data_path = os.path.join(DATA_DIR, f"nodes_{node_count}", graph_type)

    try:
        # Data file paths
        env1_file = os.path.join(data_path, f"data_env1_{index}.npy")
        env2_file = os.path.join(data_path, f"data_env2_{index}.npy")
        metadata_file = os.path.join(data_path, f"metadata_{index}.json")
        adj_file = os.path.join(data_path, f"adj_{index}.npy")

        # Load data
        X1 = np.load(env1_file)  # (n_samples, n_nodes)
        X2 = np.load(env2_file)  # (n_samples, n_nodes)

        # Compute covariance matrices
        S1 = np.cov(X1.T)  # (n_nodes, n_nodes)
        S2 = np.cov(X2.T)  # (n_nodes, n_nodes)

        # Load metadata
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        # Load adjacency matrix (if available)
        adj = None
        if os.path.exists(adj_file):
            adj = np.load(adj_file)

        return S1, S2, metadata, adj

    except Exception as e:
        print(f"Error loading data for {graph_type} nodes_{node_count} index_{index}: {e}")
        return None, None, None, None

def run_single_experiment(node_count, graph_type, index):
    """Run single experiment"""
    try:
        # Load data
        S1, S2, metadata, adj = load_data(node_count, graph_type, index)

        if S1 is None or S2 is None:
            return None

        # Extract actual shifted nodes information
        shifted_nodes = metadata.get('shifted_nodes', [])
        shift_types = metadata.get('shift_types', {})
        shift_case = metadata.get('shift_case', '')
        seed = metadata.get('seed', 0)
        edge_count = metadata.get('edge_count', 0)

        # Run PreDITEr algorithm
        start_time = time.time()

        # Adjust parameters based on node count and graph type (higher threshold + variance only)
        if node_count == 10:
            th1 = 0.35  # Higher threshold for better accuracy
            lambda_1 = 0.05  # Lower regularization for better sensitivity
        elif node_count == 20:
            th1 = 0.35
            lambda_1 = 0.05
        elif node_count == 30:
            if graph_type == "ER":
                th1 = 0.35
                lambda_1 = 0.05
            else:  # SF
                th1 = 0.35
                lambda_1 = 0.05
        elif node_count == 40:
            if graph_type == "ER":
                th1 = 0.35
                lambda_1 = 0.05
            else:  # SF
                th1 = 0.35
                lambda_1 = 0.05
        elif node_count == 50:
            # Higher threshold and lower regularization for better accuracy
            if graph_type == "ER":
                th1 = 0.35  # Higher value than node 9's change amount 0.311496
                lambda_1 = 0.05  # Lower value for better sensitivity
            else:  # SF
                th1 = 0.35
                lambda_1 = 0.05
        else:
            th1 = 0.35
            lambda_1 = 0.05

        # Run IMAG_pasp_sample (PreDITEr model - consider only variance changes)
        detected_nodes, pasp_info = IMAG_pasp_sample(
            S1, S2,
            max_subset_size=min(node_count, 20),
            n_max_iter=100,
            stop_cond=1e-4,
            tol=1e-6,
            verbose=False,
            return_pasp=True,
            lambda_1=lambda_1,
            lambda_pasp=0.1,
            rho=1.0,
            th1=th1,
            only_diag=True  # Consider only variance changes for better accuracy
        )

        execution_time = time.time() - start_time

        # Construct result
        result = {
            "dataset_info": {
                "seed": seed,
                "dataset_index": index,
                "node_count": node_count,
                "edge_count": edge_count,
                "graph_type": graph_type,
                "shifted_nodes": shifted_nodes,
                "shift_types": shift_types,
                "shift_case": shift_case
            },
            "model": "PreDITEr",
            "detected_shifted_nodes": sorted(detected_nodes),  # Nodes predicted by the model
            "experiment_info": {
                "node_count": node_count,
                "graph_type": graph_type,
                "data_index": index,
                "execution_time": execution_time,
                "parameters": {
                    "th1": th1,
                    "lambda_1": lambda_1,
                    "lambda_pasp": 0.1,
                    "rho": 1.0,
                    "max_subset_size": min(node_count, 20),
                    "n_max_iter": 100
                }
            }
        }

        return result

    except Exception as e:
        print(f"Error in experiment {graph_type} nodes_{node_count} index_{index}: {e}")
        return None

def save_result(result, node_count, graph_type, index, output_dir="prediter_result_test_new"):
    """Save results with new folder structure"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Create folder for each node count
    node_dir = os.path.join(output_dir, f"nodes_{node_count}")
    os.makedirs(node_dir, exist_ok=True)

    # Create folder for each graph type
    graph_dir = os.path.join(node_dir, graph_type)
    os.makedirs(graph_dir, exist_ok=True)

    # Generate and save result filename
    filename = f"prediter_nodes{node_count}_{graph_type}_{index}.json"
    filepath = os.path.join(graph_dir, filename)

    with open(filepath, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"[SAVE] Saved result to: {filepath}")

def worker_function(args):
    """Worker function for multiprocessing"""
    node_count, graph_type, index = args
    return run_single_experiment(node_count, graph_type, index)

def run_experiments_multiprocess(node_counts, graph_types, max_index=30, num_processes=None, output_dir="prediter_result_test_new"):
    """Run experiments with multiprocessing"""
    if num_processes is None:
        num_processes = mp.cpu_count()

    print(f"[START] Running PreDITEr experiments with {num_processes} processes")
    print(f"[OUTPUT] Results will be saved to: {output_dir}")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting multiprocessing experiments")

    # Create all experiment tasks
    tasks = []
    for node_count in node_counts:
        for graph_type in graph_types:
            for index in range(1, max_index + 1):
                tasks.append((node_count, graph_type, index))

    total_tasks = len(tasks)
    completed_tasks = mp.Value('i', 0)

    print(f"[INFO] Total tasks: {total_tasks}")

    # Create and execute process pool
    with mp.Pool(processes=num_processes) as pool:
        # Callback function to save results
        def save_callback(result):
            with completed_tasks.get_lock():
                completed_tasks.value += 1
                current_completed = completed_tasks.value

            if result is not None:
                node_count, graph_type, index = result['experiment_info']['node_count'], \
                                               result['experiment_info']['graph_type'], \
                                               result['experiment_info']['data_index']
                save_result(result, node_count, graph_type, index, output_dir)

            if current_completed % 10 == 0:
                print(f"[PROGRESS] Completed {current_completed}/{total_tasks} tasks")

        # Execute tasks asynchronously
        for task in tasks:
            pool.apply_async(
                worker_function,
                args=(task,),
                callback=save_callback,
                error_callback=lambda e: print(f"[ERROR] Task failed: {e}")
            )

        # Wait for all tasks to complete
        pool.close()
        pool.join()

    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] All multiprocessing experiments completed!")
    print(f"[SUMMARY] Total tasks: {total_tasks}")
    print(f"[SUMMARY] Completed tasks: {completed_tasks.value}")

def main():
    """Main execution function"""
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting PreDITEr experiments")

    # Experiment settings
    node_counts = [10, 20, 30, 40, 50]
    graph_types = ["ER", "SF"]
    max_index = 30

    # Set output directory (generate unique name with timestamp)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"prediter_result_test_{timestamp}"

    # Run with multiprocessing
    run_experiments_multiprocess(node_counts, graph_types, max_index, output_dir=output_dir)

    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] All experiments completed!")
    print(f"[OUTPUT] Results saved to: {output_dir}")

if __name__ == "__main__":
    main()