# FANS: Flow-based Analysis of Noise Shift

## Overview

**FANS (Flow-based Analysis of Noise Shift)** is a framework for detecting and analyzing distributional shifts in causal systems using normalizing flows. Built on the foundation of [Causal Normalizing Flows](https://proceedings.neurips.cc/paper_files/paper/2023/hash/b8402301e7f06bdc97a31bfaa653dc32-Abstract-Conference.html), FANS extends the methodology to identify whether observed distribution changes are due to functional shifts (changes in causal mechanisms) or noise shifts (changes in noise distributions).

### Key Features

- **Shift Detection**: Automatically identifies which variables have undergone distributional shifts between environments
- **Shift Type Classification**: Distinguishes between function shifts and noise shifts

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
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
```

Install additional requirements:

```bash
pip install -r requirements.txt
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

## FANS Method

The FANS (Flow-based Analysis of Noise Shift) method leverages trained causal normalizing flows to detect and classify distributional shifts between two environments.

### Core Methodology

1. **Training**: Learn a causal normalizing flow on environment 1 data that maps observations X to noise variables Z following the causal graph structure

2. **Shift Detection**: Transform environment 2 data through the learned flow and test for independence violations in the noise space

3. **Statistical Testing**: Use distance correlation and independence tests to identify shifted variables

4. **Visualization**: Generate comparative plots showing distributional differences

## Data Structure

### Synthetic Data

You can create n=50,000 synthetic data set
```
python data/generate_data.py
```


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

#### Example: Train on 30-node scale-free graph

```bash
CUDA_VISIBLE_DEVICES=1 python main.py \
    --config_file causal_nf/configs/data_small/nodes_30/SF/causal_nf_nodes_30_SF_adj_5.yaml \
    --wandb_mode online \
    --project fans_experiments
```


### Running Baseline Methods

Run baseline shift detection methods for comparison:

```bash
python experiments/experiment_script.py --model <MODEL_NAME> [OPTIONS]
```

**Available Models:**
- `splitkci`: Kernel Conditional Independence Test
- `prediter`: PreDITEr method
- `iscan`: Independence-based shift detection
- `linearccp`: Linear CCP
- `gpr`: Gaussian Process Regression


**Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `--nodes` | Node counts to process (space-separated) | 10 20 30 40 50 |
| `--gpu` | GPU device ID (-1 for CPU) | -1 |
| `--output_dir` | Results directory | auto-generated |
| `--config_type` | Graph type filter (ER, SF, all) | all |
| `--dataset_indices` | Dataset range (e.g., "1-30") | all |

#### Examples

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



