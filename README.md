# FANS: Flow-based Analysis of Noise Shift

## Overview

**FANS (Flow-based Analysis of Noise Shift)** is a framework for detecting and analyzing distributional shifts in causal systems using normalizing flows. Built on the foundation of [Causal Normalizing Flows](https://proceedings.neurips.cc/paper_files/paper/2023/hash/b8402301e7f06bdc97a31bfaa653dc32-Abstract-Conference.html), FANS extends the methodology to identify whether observed distribution changes are due to functional shifts (changes in causal mechanisms) or noise shifts (changes in noise distributions).

### Key Features

- **Shift Detection**: Automatically identifies which variables have undergone distributional shifts between environments
- **Shift Type Classification**: Distinguishes between function shifts and noise shifts
- **Baseline Comparisons**: Comprehensive comparison with state-of-the-art methods (GPR, SplitKCI, PreDITEr, ISCAN, LinearCCP)
- **Scalable Experiments**: Supports synthetic data generation and real-world datasets
- **GPU Acceleration**: Efficient training and inference on GPUs with automatic batch scheduling

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [FANS Method](#fans-method)
- [Data Structure](#data-structure)
- [Running Experiments](#running-experiments)
  - [Training FANS Model](#training-fans-model)
  - [Running Baseline Methods](#running-baseline-methods)
  - [Batch Experiments](#batch-experiments)
- [Configuration Files](#configuration-files)
- [Results and Analysis](#results-and-analysis)
- [Baseline Methods](#baseline-methods)
- [Real Data Experiments](#real-data-experiments)
- [Project Structure](#project-structure)
- [Original Causal NF Experiments](#original-causal-nf-experiments)
- [Citation](#citation)
- [Contact](#contact)

## Installation

### Prerequisites

Create a new conda environment with Python 3.9.12:

```bash
conda create --name fans python=3.9.12 --no-default-packages
```

Activate the conda environment:

```bash
conda activate fans
```

### Install Dependencies

Install PyTorch and related packages:

```bash
pip install torch==1.13.1 torchvision==0.14.1
pip install torch_geometric==2.3.1
pip install torch-scatter==2.1.1
```

Install additional requirements:

```bash
pip install -r requirements.txt
```

### Verify Installation

Run the test suite to ensure everything is set up correctly:

```bash
pytest tests/test_causal_transform.py
pytest tests/test_flows.py
pytest tests/test_scm.py
```

## Quick Start

### Train a FANS Model

Train a FANS model on a synthetic dataset with 10 nodes and Erdős-Rényi (ER) graph structure:

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
    --config_file causal_nf/configs/data_small/nodes_10/ER/causal_nf_nodes_10_ER_adj_1.yaml \
    --wandb_mode offline \
    --project causal_nf
```

**What this does:**
- Trains a causal normalizing flow on environment 1 data
- Evaluates shift detection performance on environment 2 data
- Saves results to `results/` directory
- Generates visualizations of detected shifts

**Expected outputs:**
- Trained model checkpoint
- Shift detection results (JSON format)
- Visualization plots (PDF/PNG)
- Training logs and metrics

### Run on CPU

To run without GPU:

```bash
python main.py \
    --config_file causal_nf/configs/data_small/nodes_10/ER/causal_nf_nodes_10_ER_adj_1.yaml \
    --wandb_mode offline \
    --project causal_nf \
    --opts device cpu
```

## FANS Method

The FANS (Flow-based Analysis of Noise Shift) method leverages trained causal normalizing flows to detect and classify distributional shifts between two environments.

### Core Methodology

1. **Training**: Learn a causal normalizing flow on environment 1 data that maps observations X to noise variables Z following the causal graph structure

2. **Shift Detection**: Transform environment 2 data through the learned flow and test for independence violations in the noise space

3. **Statistical Testing**: Use distance correlation and independence tests to identify shifted variables

4. **Visualization**: Generate comparative plots showing distributional differences

### Key Components

The `FANSAnalyzer` class (in `fans.py`) provides:

- **`compute_shift_statistics()`**: Computes statistical measures for shift detection
- **`detect_shifted_nodes()`**: Identifies nodes with significant distributional shifts
- **`analyze_noise_independence()`**: Tests independence assumptions in noise space
- **`generate_visualizations()`**: Creates plots for shift analysis

### Advantages

- **Causal Structure**: Exploits causal graph structure for more accurate detection
- **Shift Type Identification**: Can distinguish function shifts from noise shifts
- **Scalability**: Handles high-dimensional data efficiently
- **Theoretical Foundation**: Grounded in causal inference and normalizing flow theory

## Data Structure

### Synthetic Data

Synthetic datasets are organized by node count and graph type:

```
data/data_small/
├── nodes_10/
│   ├── ER/          # Erdős-Rényi random graphs
│   │   ├── adj_1.npy           # Adjacency matrix
│   │   ├── data_env1_1.npy     # Environment 1 data
│   │   ├── data_env2_1.npy     # Environment 2 data
│   │   └── metadata_1.json     # Shift information
│   └── SF/          # Scale-free graphs
├── nodes_20/
├── nodes_30/
├── nodes_40/
└── nodes_50/
```

### Metadata Format

Each dataset includes metadata describing the shifts:

```json
{
  "seed": 42,
  "dataset_index": 1,
  "node_count": 10,
  "edge_count": 15,
  "graph_type": "ER",
  "shifted_nodes": [3, 7],
  "shift_types": {
    "3": ["function"],
    "7": ["noise_variance"]
  },
  "shift_case": "both"
}
```

### Real Datasets

- **Morpho-MNIST**: Located in `data/morpho_mnist/`
- **Sachs**: Located in `data/sachs/`

## Running Experiments

### Training FANS Model

#### Basic Usage

```bash
python main.py \
    --config_file <CONFIG_PATH> \
    --wandb_mode <MODE> \
    --project <PROJECT_NAME>
```

**Parameters:**
- `--config_file`: Path to YAML configuration file
- `--wandb_mode`: Logging mode (`online`, `offline`, or `disabled`)
- `--project`: W&B project name for experiment tracking

#### Example: Train on 30-node scale-free graph

```bash
CUDA_VISIBLE_DEVICES=1 python main.py \
    --config_file causal_nf/configs/data_small/nodes_30/SF/causal_nf_nodes_30_SF_adj_5.yaml \
    --wandb_mode online \
    --project fans_experiments
```

#### Override Configuration Parameters

Use `--opts` to override config values:

```bash
python main.py \
    --config_file causal_nf/configs/data_small/nodes_10/ER/causal_nf_nodes_10_ER_adj_1.yaml \
    --wandb_mode offline \
    --project causal_nf \
    --opts model.num_layers 8 optim.max_epochs 500 device cuda:0
```

### Running Baseline Methods

Run baseline shift detection methods for comparison:

```bash
python experiments/experiment_script.py --model <MODEL_NAME> [OPTIONS]
```

**Available Models:**
- `gpr`: Gaussian Process Regression
- `splitkci`: Kernel Conditional Independence Test
- `prediter`: PreDITEr method
- `iscan`: Independence-based shift detection
- `linearccp`: Linear CCP

**Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `--nodes` | Node counts to process (space-separated) | 10 20 30 40 50 |
| `--gpu` | GPU device ID (-1 for CPU) | -1 |
| `--output_dir` | Results directory | auto-generated |
| `--config_type` | Graph type filter (ER, SF, all) | all |
| `--dataset_indices` | Dataset range (e.g., "1-30") | all |

#### Examples

Run GPR baseline on 10-node ER graphs using GPU 0:

```bash
python experiments/experiment_script.py \
    --model gpr \
    --nodes 10 \
    --config_type ER \
    --gpu 0
```

Run SplitKCI on all node sizes, only first 5 datasets:

```bash
python experiments/experiment_script.py \
    --model splitkci \
    --dataset_indices "1-5" \
    --gpu 0
```

Run ISCAN on CPU for SF graphs:

```bash
python experiments/experiment_script.py \
    --model iscan \
    --config_type SF \
    --gpu -1
```

### Batch Experiments

For running multiple experiments in parallel across GPUs, use the batch scheduler:

```bash
bash run_experiments.sh
```

**Configuration** (edit `run_experiments.sh`):

```bash
NODE_SIZE="nodes_50"  # Node size for experiments
GRAPH_TYPE="SF"       # Graph type (ER or SF)
```

**Features:**
- Automatically distributes experiments across GPUs 0-5
- Creates tmux sessions for each experiment
- 5 experiments per GPU (30 total for datasets 1-30)
- Monitors and logs all experiments

**Monitoring:**

List running experiments:
```bash
tmux ls
```

Attach to a specific experiment:
```bash
tmux attach -t gpu0_nodes_50_SF_1
```

Monitor GPU usage:
```bash
watch -n 5 nvidia-smi
```

Kill all experiments:
```bash
tmux kill-server
```

## Configuration Files

Configuration files are located in `causal_nf/configs/` and use YAML format.

### Base Configuration

`default_config.yaml` contains default settings for:
- Model architecture (flow type, number of layers, hidden dimensions)
- Training parameters (learning rate, batch size, epochs)
- Data loading settings
- Logging and visualization options

### Experiment-Specific Configs

Located in `causal_nf/configs/data_small/nodes_X/{ER,SF}/`:

```yaml
# Example: causal_nf_nodes_10_ER_adj_1.yaml
dataset:
  name: synthetic
  num_nodes: 10
  graph_type: ER
  dataset_index: 1
  data_path: data/data_small/nodes_10/ER/
  env1_data: data_env1_1.npy
  env2_data: data_env2_1.npy
  adjacency_path: adj_1.npy
  metadata_path: metadata_1.json

model:
  type: causal_nf
  num_layers: 6
  hidden_dims: [64, 64]
  flow_type: nsf  # Neural Spline Flow

optim:
  optimizer: adam
  base_lr: 0.001
  max_epochs: 300
  batch_size: 256
```

### Key Parameters

**Model Parameters:**
- `model.num_layers`: Number of flow transformation layers
- `model.hidden_dims`: Hidden layer dimensions for neural networks
- `model.flow_type`: Type of flow (`nsf`, `maf`, `coupling`)

**Training Parameters:**
- `optim.base_lr`: Learning rate
- `optim.max_epochs`: Maximum training epochs
- `optim.batch_size`: Batch size
- `optim.early_stopping`: Enable early stopping

**Data Parameters:**
- `dataset.data_path`: Path to dataset directory
- `dataset.train_ratio`: Train/validation split ratio
- `dataset.standardize`: Whether to standardize inputs

## Results and Analysis

### Analysis Scripts

Located in `experiments/analysis/`:

**`analyze_both.py`**: Compare FANS with all baseline methods

```bash
python experiments/analysis/analyze_both.py
```

Generates:
- Comparative performance tables
- ROC curves and precision-recall plots
- Statistical significance tests
- Aggregated results across all datasets

### Results Directory Structure

```
experiments/results/
├── both/                    # Combined FANS + baseline results
├── gpr_<timestamp>/        # GPR results
├── splitkci/               # SplitKCI results
├── prediter_n=<sample>/    # PreDITEr results
├── iscan/                  # ISCAN results
└── linearccp_n=<sample>/   # LinearCCP results
```

### Output Format

Each method produces JSON results:

```json
{
  "dataset_info": {
    "node_count": 10,
    "graph_type": "ER",
    "dataset_index": 1,
    "true_shifted_nodes": [3, 7]
  },
  "detected_shifted_nodes": [3, 7, 9],
  "detection_metrics": {
    "precision": 0.67,
    "recall": 1.0,
    "f1_score": 0.80
  },
  "execution_time": 45.2
}
```

### Unified Analysis

Generate unified results CSV:

```bash
python experiments/analysis/analyze_both.py --output unified_analysis_results.csv
```

## Baseline Methods

### GPR (Gaussian Process Regression)

Detects shifts by comparing GP models fitted on parent-child relationships in different environments.

**Location:** `experiments/baselines/gpr.py`

**Method:** Tests if conditional distributions P(Y|Parents) differ between environments using likelihood ratio tests.

### SplitKCI (Kernel Conditional Independence Test)

Uses kernel-based conditional independence testing to detect environment-dependent relationships.

**Location:** `experiments/baselines/splitkci.py`

**Method:** Tests if Y ⊥ Z | Parents, where Z is the environment indicator.

### PreDITEr (Predicting Distributional Shifts)

Detects shifts by analyzing changes in covariance structure between environments.

**Location:** `experiments/baselines/prediter/`

**Method:** Uses sparse optimization to identify nodes with variance or covariance changes.

### ISCAN (Independence-based Shift Detection)

Identifies shifts through independence testing in estimated noise distributions.

**Location:** `experiments/baselines/iscan.py`

**Method:** Tests independence of estimated noise terms between environments.

### LinearCCP (Linear Causal Consistency Principle)

Assumes linear causal relationships and detects violations of causal consistency.

**Location:** `experiments/baselines/linearccp.py`

**Method:** Linear regression-based approach for shift detection in linear SEMs.

## Real Data Experiments

### Morpho-MNIST Dataset

The Morpho-MNIST dataset contains handwritten digits with controlled morphological variations.

**Data location:** `data/morpho_mnist/`

**Run FANS on Morpho-MNIST:**

```bash
python main.py \
    --config_file causal_nf/configs/morpho_mnist.yaml \
    --wandb_mode online \
    --project fans_morpho
```

**Analysis:**

```bash
jupyter notebook experiments/analysis/morpho_mnist.ipynb
```

### Sachs Dataset

Real-world protein signaling data with known causal relationships.

**Data location:** `data/sachs/`

**Run FANS on Sachs:**

```bash
python main.py \
    --config_file causal_nf/configs/sachs.yaml \
    --wandb_mode online \
    --project fans_sachs
```

**Features:**
- 11 phosphoproteins and phospholipids
- Multiple experimental conditions (observational and interventional)
- Known ground-truth causal graph

## Project Structure

```
fans/
├── causal_nf/                      # Core causal NF implementation
│   ├── configs/                    # Configuration files
│   │   ├── default_config.yaml    # Base configuration
│   │   └── data_small/            # Experiment configs
│   ├── datasets/                   # Dataset classes
│   ├── distributions/              # Probability distributions
│   ├── models/                     # Model implementations
│   │   ├── base_model.py
│   │   └── causal_nf.py
│   ├── modules/                    # Neural network modules
│   ├── preparators/                # Data preparators
│   ├── sem_equations/              # Structural equation models
│   ├── transforms/                 # Flow transformations
│   └── utils/                      # Utility functions
├── data/                           # Experimental datasets
│   ├── data_small/                 # Synthetic data
│   │   └── nodes_{10,20,30,40,50}/
│   │       ├── ER/                 # Erdős-Rényi graphs
│   │       └── SF/                 # Scale-free graphs
│   ├── morpho_mnist/               # Morpho-MNIST dataset
│   └── sachs/                      # Sachs protein data
├── experiments/                    # Baseline methods and analysis
│   ├── baselines/                  # Baseline implementations
│   │   ├── gpr.py
│   │   ├── splitkci.py
│   │   ├── prediter/
│   │   ├── iscan.py
│   │   └── linearccp.py
│   ├── analysis/                   # Analysis scripts
│   │   ├── analyze_both.py
│   │   └── morpho_mnist.ipynb
│   ├── experiment_script.py        # Baseline runner
│   └── results/                    # Experimental results
├── zuko/                           # Normalizing flow library (modified)
├── torchlikelihoods/              # Likelihood implementations
├── fans.py                        # FANS analyzer implementation
├── main.py                        # Main training script
├── visualize.py                   # Visualization utilities
├── run_experiments.sh             # Batch experiment scheduler
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Original Causal NF Experiments

This repository also contains code to reproduce experiments from the original ["Causal Normalizing Flows: From Theory to Practice"](https://proceedings.neurips.cc/paper_files/paper/2023/hash/b8402301e7f06bdc97a31bfaa653dc32-Abstract-Conference.html) paper.

### Causal Graphs of SCMs

Generate plots of causal graphs:

```bash
pytest tests/test_scm_plots.py -k test_scm_plot_graph
```

### Ablation Studies

#### Flow Direction Ablation

```bash
python generate_jobs.py --grid_file grids/causal_nf/ablation_u_x/base.yaml --format shell --jobs_per_file 20000 --batch_size 4
python generate_jobs.py --grid_file grids/causal_nf/ablation_x_u/base.yaml --format shell --jobs_per_file 20000 --batch_size 4

# Generate figures
python scripts/create_figure_ablation_direction.py
python scripts/create_figure_ablation_best.py
python scripts/create_figure_ablation_time.py
```


## Citation

If you use this code or the FANS method in your work, please cite:

### FANS Extension

```
@article{fans2024,
    title={FANS: Flow-based Analysis of Noise Shift for Distributional Shift Detection},
    author={[Your Name/Team]},
    year={2024}
}
```

### Original Causal Normalizing Flows

```
@inproceedings{javaloy2023causal,
    title={Causal normalizing flows: from theory to practice},
    author={Adri{\'a}n Javaloy and Pablo Sanchez Martin and Isabel Valera},
    booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
    year={2023},
    url={https://openreview.net/forum?id=QIFoCI7ca1}
}
```

## License

This project builds upon the Causal Normalizing Flows codebase. The normalizing flow implementation in the `zuko` folder is a modified version of [Zuko v0.2.0](https://github.com/probabilists/zuko/releases/tag/0.2.0) (MIT license).

For future work, please consider using the latest version of Zuko instead of the included version.

## Contact

For questions, feedback, or inquiries:

**Original Causal NF Authors:**
- Pablo Sanchez Martin: [psanchez@tue.mpg.de](mailto:psanchez@tue.mpg.de)
- Adrián Javaloy: [ajavaloy@cs.uni-saarland.de](mailto:ajavaloy@cs.uni-saarland.de)

**FANS Extension:**
- [Add your contact information here]

For issues related to the repository or code, create a GitHub issue or pull request.

---

We appreciate your interest in our research and code! Your feedback and collaboration are valuable to us.
