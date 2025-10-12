#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script generates synthetic data using the updated DataGenerator class and saves it to a data folder.
Uses random ER or SF graphs and applies different types of shifts to selected nodes.
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
NODE_COUNTS = [10, 20, 30, 40, 50]  # Node counts
GRAPH_TYPES = ["ER", "SF"]  # Graph types

# New 6 shift cases
SHIFT_CASES = [
    DataGenerator.SHIFT_CASE1,  # sin->cos, N(0,2)
    DataGenerator.SHIFT_CASE2,  # sin->cos, Laplace(0,1)
    DataGenerator.SHIFT_CASE3,  # PA_j delete, N(0,2)
    DataGenerator.SHIFT_CASE4,  # PA_j delete, Laplace(0,1)
    DataGenerator.SHIFT_CASE5,  # sin->cos, PA_j delete
    DataGenerator.SHIFT_CASE6,  # N(0,2), Laplace(0,1)
]
    
def get_shift_cases_for_node_count(node_count):
    cases = []
    for case in SHIFT_CASES:
        cases.extend([case] * 5)
    return cases

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
    color_sin_cos = "red"
    color_pa_delete = "orange"
    color_noise_n02 = "green"
    color_noise_laplace = "purple"

    # Assign colors and labels to nodes
    colors = []
    labels = []
    
    for i in range(gen.d):
        if i in gen.sin_cos_shift_nodes:
            c = color_sin_cos
        elif i in gen.pa_delete_shift_nodes:
            c = color_pa_delete
        elif i in gen.noise_n02_shift_nodes:
            c = color_noise_n02
        elif i in gen.noise_laplace_shift_nodes:
            c = color_noise_laplace
        else:
            c = color_no_shift
            
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
            types.append("sin_cos")
        if node in gen.pa_delete_shift_nodes:
            types.append("pa_delete")
        if node in gen.noise_n02_shift_nodes:
            types.append("noise_N02")
        if node in gen.noise_laplace_shift_nodes:
            types.append("noise_laplace")
        shift_types[int(node)] = types
    
    # Create metadata
    metadata = {
        "seed": seed,
        "dataset_index": dataset_idx,
        "node_count": gen.d,
        "edge_count": gen.s0,
        "graph_type": gen.graph_type,
        "shifted_nodes": gen.shifted_nodes,
        "shift_types": shift_types,
        "shift_case": gen.shift_case,
        "deleted_parents": gen.deleted_parents
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
    print(f"Used shift cases: {SHIFT_CASES}")
    print("Dataset counts per configuration:")
    for node_count in NODE_COUNTS:
        cases = get_shift_cases_for_node_count(node_count)
        print(f"  {node_count} nodes: {len(cases)} datasets")

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
                
                # Define number of shifted nodes
                if node_count == 10:
                    num_shifted = 2  # 2 nodes for 10-node graphs
                elif node_count == 20:
                    num_shifted = 4  # 4 nodes for 20-node graphs
                else:
                    # For larger graphs, use 20% of nodes, ensuring even number
                    num_shifted = max(2, int(node_count * 0.2))
                    if num_shifted % 2 != 0:
                        num_shifted += 1
                
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