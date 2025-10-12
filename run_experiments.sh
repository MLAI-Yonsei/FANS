#!/bin/bash

# ===============================
# Static Experiment Assignment Scheduler per GPU (GPU 1-6, 5 experiments each, ER only)
# ===============================

set -e

# Common settings
CONDA_ENV="fans"
WORK_DIR="/home/statduck/fans"
PROJECT_NAME="causal_nf"

# Experiment settings (modifiable)
NODE_SIZE="nodes_50"  # Node size for experiments
GRAPH_TYPE="SF"       # Graph type (SF only)

echo "================================================"
echo "Starting static experiment assignment per GPU!"
echo "Node size: $NODE_SIZE"
echo "Graph type: $GRAPH_TYPE"
echo "GPU range: 0-5 (5 experiments per GPU)"
echo "================================================"

# Function to extract wandb group
get_wandb_group() {
    local config_file=$1
    echo $(dirname "$config_file")
}

# Assign task to GPU
assign_task_to_gpu() {
    local gpu_id=$1
    local config_file=$2
    local exp_name=$3
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Assigning task to GPU $gpu_id: $exp_name"
    
    # Create wandb group
    local wandb_group=$(get_wandb_group "$config_file")
    
    # tmux session name
    local session_name="gpu${gpu_id}_${exp_name}"
    
    # Kill existing session if any
    tmux kill-session -t "$session_name" 2>/dev/null || true
    
    # Create tmux session and run experiment
    tmux new-session -d -s "$session_name" -c "$WORK_DIR"
    tmux send-keys -t "$session_name" "conda activate $CONDA_ENV" Enter
    tmux send-keys -t "$session_name" "cd $WORK_DIR" Enter
    tmux send-keys -t "$session_name" "
        echo 'GPU $gpu_id: Starting experiment $exp_name'
        CUDA_VISIBLE_DEVICES=$gpu_id python main.py \\
            --config_file causal_nf/configs/data_small/$config_file \\
            --wandb_mode online \\
            --project $PROJECT_NAME \\
            --wandb_group '$wandb_group'
        echo 'GPU $gpu_id: Experiment $exp_name completed'
        exit
    " Enter
    
    return 0
}

# GPU 0: ER 1-5
echo ""
echo "=== GPU 0: ${GRAPH_TYPE} 1-5 assignment start ==="
for i in {1..5}; do
    config_file="${NODE_SIZE}/${GRAPH_TYPE}/causal_nf_${NODE_SIZE}_${GRAPH_TYPE}_adj_${i}.yaml"
    exp_name="${NODE_SIZE}_${GRAPH_TYPE}_${i}"
    
    assign_task_to_gpu 0 "$config_file" "$exp_name"
    sleep 2  # Interval between consecutive assignments
done
echo "GPU 0 assignment complete!"

# GPU 1: ER 6-10
echo ""
echo "=== GPU 1: ${GRAPH_TYPE} 6-10 assignment start ==="
for i in {6..10}; do
    config_file="${NODE_SIZE}/${GRAPH_TYPE}/causal_nf_${NODE_SIZE}_${GRAPH_TYPE}_adj_${i}.yaml"
    exp_name="${NODE_SIZE}_${GRAPH_TYPE}_${i}"
    
    assign_task_to_gpu 1 "$config_file" "$exp_name"
    sleep 2  # Interval between consecutive assignments
done
echo "GPU 1 assignment complete!"

# GPU 2: ER 11-15
echo ""
echo "=== GPU 2: ${GRAPH_TYPE} 11-15 assignment start ==="
for i in {11..15}; do
    config_file="${NODE_SIZE}/${GRAPH_TYPE}/causal_nf_${NODE_SIZE}_${GRAPH_TYPE}_adj_${i}.yaml"
    exp_name="${NODE_SIZE}_${GRAPH_TYPE}_${i}"
    
    assign_task_to_gpu 2 "$config_file" "$exp_name"
    sleep 2  # Interval between consecutive assignments
done
echo "GPU 2 assignment complete!"

# GPU 3: ER 16-20
echo ""
echo "=== GPU 3: ${GRAPH_TYPE} 16-20 assignment start ==="
for i in {16..20}; do
    config_file="${NODE_SIZE}/${GRAPH_TYPE}/causal_nf_${NODE_SIZE}_${GRAPH_TYPE}_adj_${i}.yaml"
    exp_name="${NODE_SIZE}_${GRAPH_TYPE}_${i}"
    
    assign_task_to_gpu 3 "$config_file" "$exp_name"
    sleep 2  # Interval between consecutive assignments
done
echo "GPU 3 assignment complete!"

# GPU 4: ER 21-25
echo ""
echo "=== GPU 4: ${GRAPH_TYPE} 21-25 assignment start ==="
for i in {21..25}; do
    config_file="${NODE_SIZE}/${GRAPH_TYPE}/causal_nf_${NODE_SIZE}_${GRAPH_TYPE}_adj_${i}.yaml"
    exp_name="${NODE_SIZE}_${GRAPH_TYPE}_${i}"
    
    assign_task_to_gpu 4 "$config_file" "$exp_name"
    sleep 2  # Interval between consecutive assignments
done
echo "GPU 4 assignment complete!"

# GPU 5: ER 26-30
echo ""
echo "=== GPU 5: ${GRAPH_TYPE} 26-30 assignment start ==="
for i in {26..30}; do
    config_file="${NODE_SIZE}/${GRAPH_TYPE}/causal_nf_${NODE_SIZE}_${GRAPH_TYPE}_adj_${i}.yaml"
    exp_name="${NODE_SIZE}_${GRAPH_TYPE}_${i}"
    
    assign_task_to_gpu 5 "$config_file" "$exp_name"
    sleep 2  # Interval between consecutive assignments
done
echo "GPU 5 assignment complete!"

echo ""
echo "================================================"
echo "Task assignment to all GPUs complete!"
echo "Total 30 experiments (${NODE_SIZE} ${GRAPH_TYPE}: GPU 0-5, 5 each)"
echo "================================================"

# Check running sessions
echo ""
echo "Currently running tmux sessions:"
tmux ls | grep -E "gpu[0-5]_" || echo "  No sessions"

# Cleanup function
cleanup() {
    echo ""
    echo "================================================"
    echo "Scheduler terminated"
    echo "================================================"
    
    echo "Running tmux sessions:"
    tmux ls | grep -E "gpu[0-5]_" || echo "  No running sessions"
    
    echo ""
    echo "To kill all sessions:"
    echo "  tmux kill-server"
    echo ""
    echo "To check individual sessions:"
    echo "  tmux ls"
    echo "  tmux attach -t <session_name>"
}

# Signal handling
trap cleanup EXIT

echo ""
echo "================================================"
echo "Experiment monitoring commands:"
echo "================================================"
echo "tmux ls                          # List all sessions"
echo "tmux attach -t <session_name>    # Attach to specific session"
echo "watch -n 5 nvidia-smi            # Monitor GPU"
echo "tmux kill-server                 # Kill all sessions"
echo ""

# Script termination (only assign, no monitoring)
echo "All experiments are running in the background."
echo "Check experiment progress with 'tmux ls' and 'nvidia-smi'."