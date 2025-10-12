import os
import json
import numpy as np
import torch
import time
from datetime import datetime
from splitkci.dependence_measures import KCIMeasure

# GPU settings and seed initialization
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
torch.manual_seed(42)
np.random.seed(42)

# Load data
def load_data(adj_path, env1_path, env2_path, metadata_path):
    adj_matrix = np.load(adj_path)
    data_env1 = np.load(env1_path)
    data_env2 = np.load(env2_path)
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    shifted_nodes = metadata.get("shifted_nodes", [])
    return adj_matrix, data_env1, data_env2, shifted_nodes, metadata

# Identify parent-child relationships
def get_parent_child_relationships(adj_matrix):
    parent_child_pairs = []
    for child in range(adj_matrix.shape[0]):
        parents = np.where(adj_matrix[:, child] == 1)[0]
        if len(parents) > 0:
            parent_child_pairs.append((child, parents))
    return parent_child_pairs

# Detect shifts
def detect_shifts(parent_child_pairs, data_env1, data_env2):
    detected_shifts = []

    for child, parents in parent_child_pairs:
        # X = parents, Y = child, Z = environment (0/1)
        # Goal: Check if Y ⟂ Z | X holds → if p-value < 0.05, Y|X varies by environment ⇒ shift detected

        # Parent (X) and child (Y) data (two environments)
        parent_data_env1 = data_env1[:, parents]
        parent_data_env2 = data_env2[:, parents]
        child_data_env1 = data_env1[:, child]
        child_data_env2 = data_env2[:, child]

        # Combine along sample axis (2n, ·)
        parent_data = np.vstack([parent_data_env1, parent_data_env2])
        child_data = np.hstack([child_data_env1, child_data_env2])
        env_labels = np.array([0] * len(child_data_env1) + [1] * len(child_data_env2))

        # Adjust dimensions if necessary
        if env_labels.ndim == 1:
            env_labels = env_labels[:, None]
        if child_data.ndim == 1:
            child_data = child_data[:, None]

        # Convert to tensors
        parent_data_t = torch.tensor(parent_data, dtype=torch.float32)
        child_data_t = torch.tensor(child_data, dtype=torch.float32)
        env_labels_t = torch.tensor(env_labels, dtype=torch.float32)

        # Kernel parameters (simply use variance; environment Z is binary so fixed value is also possible)
        sigma_x = float(parent_data_t.var().item() + 1e-8)
        sigma_y = float(child_data_t.var().item() + 1e-8)
        sigma_z = float(env_labels_t.var().item() + 1e-8)  # ≈ 0.25 (when balanced)

        kernel_a_args = {"sigma2": sigma_y}       # a = Y
        kernel_b_args = {"sigma2": sigma_z}       # b = Z
        kernel_ca_args = {"sigma2": sigma_x}      # X kernel for c -> a regression
        kernel_cb_args = {"sigma2": sigma_x}      # X kernel for c -> b regression

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

        # Order: test(a=Y, b=Z, c=X) → Y ⟂ Z | X test
        p_value = kci.test(child_data_t, env_labels_t, parent_data_t, train_test_split=0.001)

        if p_value < 0.05:
            detected_shifts.append(child)

    return detected_shifts

# Helper function to make objects JSON serializable
def make_json_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        return float(obj)
    elif isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    elif isinstance(obj, dict):
        return {str(k): make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple, set)):
        return [make_json_serializable(item) for item in obj]
    else:
        return obj

# Iterate over all cases
data_dir = "data"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
main_results_dir = f"results/SplitKCI_{timestamp}"
os.makedirs(main_results_dir, exist_ok=True)

# Define node counts and noise types
node_counts = [40, 50]
noise_types = ["ER", "SF"]

# Process each node count
for node_count in node_counts:
    print(f"\n===== Processing experiments with {node_count} nodes =====")
    
    # Create results directory for this node count
    node_results_dir = f"{main_results_dir}/nodes_{node_count}"
    os.makedirs(node_results_dir, exist_ok=True)
    
    # Process each noise type
    for noise_type in noise_types:
        print(f"\nNoise Type: {noise_type}")
        
        # Create results directory for this noise type
        noise_results_dir = f"{node_results_dir}/{noise_type}"
        os.makedirs(noise_results_dir, exist_ok=True)
        
        # Path to data for this node count and noise type
        data_path = f"{data_dir}/nodes_{node_count}/{noise_type}"
        
        # Check if this path exists
        if not os.path.exists(data_path):
            print(f"WARNING: Data path {data_path} does not exist. Skipping...")
            continue
        
        # Find all dataset indices (extract from filenames)
        dataset_indices = [
            int(file.replace("metadata_", "").replace(".json", ""))
            for file in os.listdir(data_path) if file.startswith("metadata_") and file.endswith(".json")
        ]
        dataset_indices.sort()
        
        # Process each dataset
        for dataset_idx in dataset_indices:
            
            print(f"\nProcessing dataset {dataset_idx}")
            
            # Load data and metadata
            adj_path = f"{data_path}/adj_{dataset_idx}.npy"
            env1_path = f"{data_path}/data_env1_{dataset_idx}.npy"
            env2_path = f"{data_path}/data_env2_{dataset_idx}.npy"
            metadata_path = f"{data_path}/metadata_{dataset_idx}.json"
            
            adj_matrix, data_env1, data_env2, shifted_nodes, metadata = load_data(adj_path, env1_path, env2_path, metadata_path)
            
            parent_child_pairs = get_parent_child_relationships(adj_matrix)

            # Record shift detection start time
            detection_start_time = time.time()
            
            # Perform shift detection
            detected_shifts = detect_shifts(parent_child_pairs, data_env1, data_env2)
            
            # Record shift detection end time
            detection_end_time = time.time()
            detection_time = detection_end_time - detection_start_time

            # Save results
            result = {
                "dataset_info": metadata,
                "detected_shifts": detected_shifts,
                "shifted_nodes": shifted_nodes,
                "processing_time": {
                    "detection_time_seconds": detection_time,
                    "num_parent_child_pairs": len(parent_child_pairs)
                }
            }

            # Save results for this dataset
            result_filename = f"results_{dataset_idx}.json"
            result_filepath = os.path.join(noise_results_dir, result_filename)
            with open(result_filepath, "w") as f:
                json.dump(make_json_serializable(result), f, indent=2)
            
            print(f"Dataset {dataset_idx} processing completed:")
            print(f"  - Detection time: {detection_time:.2f} seconds")
            print(f"  - Parent-child pairs: {len(parent_child_pairs)}")
            print(f"Results saved to {result_filepath}")

print(f"\n===== All experiments completed =====")
print(f"Results saved to {main_results_dir}/")