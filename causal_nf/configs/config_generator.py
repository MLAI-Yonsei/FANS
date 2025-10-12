import os

def create_config_structure():
    # Directory structure
    nodes = [10,20,30,40,50]
    graph_types = ['ER','SF']
    base_configs_dir = ''
    
    # Create directory structure and config files
    for node_count in nodes:
        for graph_type in graph_types:
            # Create directory
            dir_path = os.path.join(base_configs_dir, f'nodes_{node_count}', graph_type)
            os.makedirs(dir_path, exist_ok=True)
            print(f"Created directory: {dir_path}")
            
            # Generate config files for adj_1 to adj_30
            for adj_idx in range(1, 31):
                # Create config content as string (matching original format exactly)
                config_content = f"""device: cuda
root_dir: results/synthetic
seed: 1
dataset:
  root: ../Data
  name: null
  sem_name: null
  env2_name: null
  env2_sem_name: null
  env2_base_version: null
  splits: [ 0.8, 0.2, 0.0 ]
  k_fold: 1
  shuffle_train: True
  loss: default
  scale: default
  num_samples: 1000
  base_version: null
  type: torch
  use_edge_attr: False
  external_dag_path: data/data_small/nodes_{node_count}/{graph_type}/adj_{adj_idx}.npy
  external_data_path: data/data_small/nodes_{node_count}/{graph_type}/data_env1_{adj_idx}.npy
  env2_external_data_path: data/data_small/nodes_{node_count}/{graph_type}/data_env2_{adj_idx}.npy
  test_data_path: data/data_small/nodes_{node_count}/{graph_type}/data_env1_{adj_idx}.npy
  env2_test_data_path: data/data_small/nodes_{node_count}/{graph_type}/data_env2_{adj_idx}.npy
  metadata_path: data/data_small/nodes_{node_count}/{graph_type}/metadata_{adj_idx}.json
model:
  name: causal_nf
  layer_name: naf
  dim_inner: [32, 32, 32]
  num_layers: 1
  init: None
  act: elu
  adjacency: True
  base_to_data: False
  base_distr: normal
  learn_base: False
  plot: False
train:
  max_epochs: 3000
  regularize: False
  kl: forward  # backward
  batch_size: 1024
  num_workers: 0
  limit_train_batches: None
  limit_val_batches: None
  max_time: 00:01:00:00
  inference_mode: False
optim:
  optimizer: adam
  base_lr: 0.01
  beta_1: 0.9
  beta_2: 0.999
  momentum: 0.0
  weight_decay: 0.0
  scheduler: plateau
  mode: min
  factor: 0.95
  patience: 60
  cooldown: 0
"""
                
                # Create filename
                filename = f'causal_nf_nodes_{node_count}_{graph_type}_adj_{adj_idx}.yaml'
                filepath = os.path.join(dir_path, filename)
                
                # Write config file
                with open(filepath, 'w') as f:
                    f.write(config_content)
                
                print(f"Created config: {filepath}")

if __name__ == "__main__":
    create_config_structure()
    print("Config structure creation completed!")
    print("Total files created: 5 nodes × 2 graph types × 30 configs = 300 config files")