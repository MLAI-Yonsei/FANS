#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script generates synthetic data using the updated DataGenerator class and saves it to a data folder.
Uses random ER or SF graphs and applies different types of complete transformations to selected nodes.
"""

import os
import numpy as np
import json
import matplotlib.pyplot as plt
from datagenerator import DataGenerator, set_seed
import argparse
import random
import igraph as ig
import torch

# Constants
DATA_DIR = "data"
NODE_COUNTS = [10]  # Only 10 nodes
GRAPH_TYPES = ["ER"]  # Only ER graphs

# Three shift cases
# 15 datasets: Function + Variance shift (one node function only, one node function+variance)
# 15 datasets: Function + Distribution shift (one node function only, one node function+distribution)
SHIFT_CASES = (
    [DataGenerator.FUNC_NOISE_VAR] * 15 +      # 15 function + variance
    [DataGenerator.FUNC_NOISE_DIST] * 15       # 15 function + distribution
)  # Total: 30 datasets
    
def get_shift_cases_for_node_count(node_count):
    """Get all 30 shift cases."""
    return SHIFT_CASES

def json_serializer(obj):
    """JSON serializer for objects not serializable by default json code"""
    if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Type {type(obj)} not serializable")

def create_directories():
    """Create the necessary directories for data storage."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

def create_dataset_paths(node_count, graph_type):
    """Create directory paths for dataset storage."""
    node_data_dir = f"{DATA_DIR}/nodes_{node_count}"
    if not os.path.exists(node_data_dir):
        os.makedirs(node_data_dir)
    
    graph_data_dir = f"{node_data_dir}/{graph_type}"
    if not os.path.exists(graph_data_dir):
        os.makedirs(graph_data_dir)
    
    return graph_data_dir

def plot_dag_with_shifts(gen, filename):
    """
    Create and save a visualization of the DAG with shifted nodes highlighted.
    
    Args:
        gen: DataGenerator instance
        filename: Output path for the plot
    """
    adj = gen.adj_env1.copy()
    g = ig.Graph.Adjacency(adj.tolist(), mode="directed")
    layout = g.layout("kk")  # Kamada-Kawai layout

    # Define colors for different types of shifts
    color_no_shift = "lightblue"
    color_func_only = "yellow"
    color_func_var = "orange"
    color_func_dist = "red"

    # Assign colors and labels to nodes
    colors = []
    labels = []
    
    for i in range(gen.d):
        # Check shift combinations
        if i in gen.sin_cos_shift_nodes and i in gen.noise_var_shift_nodes:
            c = color_func_var  # Function + Variance
        elif i in gen.sin_cos_shift_nodes and i in gen.noise_dist_shift_nodes:
            c = color_func_dist  # Function + Distribution
        elif i in gen.sin_cos_shift_nodes:
            c = color_func_only  # Function only
        else:
            c = color_no_shift  # No shift
            
        lbl = str(i)
        colors.append(c)
        labels.append(lbl)

    g.vs["color"] = colors
    g.vs["label"] = labels
    g.vs["size"] = 30

    ig.plot(g,
           target=filename,
           layout=layout,
           bbox=(600, 400),
           margin=50,
           vertex_label_color="black")

def save_dataset_files(gen, data_env1, data_env2, config_data_dir, dataset_idx, seed):
    """Save dataset files including data, plots, and metadata."""
    # Save DAG image
    dag_filename = f"{config_data_dir}/dag_{dataset_idx}.png"
    plot_dag_with_shifts(gen, filename=dag_filename)
    
    # Save data arrays and adjacency matrix
    np.save(f"{config_data_dir}/data_env1_{dataset_idx}.npy", data_env1)
    np.save(f"{config_data_dir}/data_env2_{dataset_idx}.npy", data_env2)
    np.save(f"{config_data_dir}/adj_{dataset_idx}.npy", gen.adj_env1)
    
    # Create shift type mappings
    shift_types = {}
    for node in gen.shifted_nodes:
        types = []
        if node in gen.sin_cos_shift_nodes:
            types.append("function_shift")
        if node in gen.noise_var_shift_nodes:
            types.append("variance_shift")
        if node in gen.noise_dist_shift_nodes:
            types.append("distribution_shift")
        shift_types[int(node)] = types
    
    # Create metadata
    metadata = {
        "seed": seed,
        "dataset_index": dataset_idx,
        "node_count": gen.d,
        "edge_count": gen.s0,
        "graph_type": gen.graph_type,
        "shifted_nodes": gen.shifted_nodes.tolist() if hasattr(gen.shifted_nodes, 'tolist') else list(gen.shifted_nodes),
        "shift_types": shift_types,
        "shift_case": gen.shift_case,
    }
    
    # Save metadata with custom serializer
    with open(f"{config_data_dir}/metadata_{dataset_idx}.json", 'w') as f:
        json.dump(metadata, f, indent=2, default=json_serializer)

def print_generation_summary():
    """Print summary information about the data generation process."""
    print("\n===== Data generation completed =====")
    print(f"All datasets saved to {DATA_DIR}/")
    print(f"Generated data for node counts: {NODE_COUNTS}")
    print(f"Generated data for graph types: {GRAPH_TYPES}")
    print(f"\nShift case distribution (2 shifted nodes per dataset):")
    print(f"  - Function + Variance shift: 15 datasets")
    print(f"    (Node 1: function shift only, Node 2: function + variance shift N(0,1)->N(0,2))")
    print(f"  - Function + Distribution shift: 15 datasets")
    print(f"    (Node 1: function shift only, Node 2: function + distribution shift N(0,1)->Laplace(0,1))")
    print(f"  - Total: 30 datasets")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='Base seed for data generation')
    return parser.parse_args()

def main():
    """Main function to generate datasets."""
    args = parse_args()
    base_seed = args.seed
    
    # Create base directories
    create_directories()
    
    # Generate data for each node count and graph type
    dataset_counter = 0
    
    for node_count in NODE_COUNTS:
        print(f"\n===== Generating data with {node_count} nodes =====")
        
        # Get shift cases for this node count
        shift_cases = get_shift_cases_for_node_count(node_count)
        
        for graph_type in GRAPH_TYPES:
            print(f"\nGraph Type: {graph_type}")
            config_dir = create_dataset_paths(node_count, graph_type)
            
            for dataset_idx, shift_case in enumerate(shift_cases, 1):
                dataset_seed = base_seed + dataset_counter
                dataset_counter += 1
                
                progress_info = f"Dataset {dataset_idx}/{len(shift_cases)} for {graph_type} ({node_count} nodes)"
                print(f"{progress_info} - Shift Case: {shift_case}")
                
                # Set seed for reproducibility
                set_seed(dataset_seed)
                
                # Define number of shifted nodes (2 nodes for 10-node graphs)
                num_shifted = 2
                
                # Create generator instance with specified shift case
                gen = DataGenerator(
                    d=node_count,
                    s0=node_count,  # Use node count for edge count
                    graph_type=graph_type,
                    shift_case=shift_case
                )
                
                # Generate data with specified number of shifted nodes
                data_env1, data_env2 = gen.sample(n=50000, num_shifted_nodes=num_shifted)
                
                # Save dataset files
                save_dataset_files(gen, data_env1, data_env2, config_dir, dataset_idx, dataset_seed)
    
    # Print summary
    print_generation_summary()

if __name__ == "__main__":
    main() 