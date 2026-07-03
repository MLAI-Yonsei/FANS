#!/bin/bash

# ===============================
# Baseline Model Parallel Runner (ISCAN, SplitKCI, LinearCCP)
# ===============================

set -e

# Parse arguments
MODEL="${1:-iscan}"  # iscan, splitkci, linearccp
NODE_SIZE="${2:-10}"
GRAPH_TYPE="${3:-ER}"
NUM_DATASETS="${4:-30}"

# Common settings
CONDA_ENV="fans"
WORK_DIR="/home/statduck/fans/experiments"
OUTPUT_DIR="$WORK_DIR/costs"  # 결과 저장 경로
DATA_DIR="/mlainas/statduck/data_small"

echo "================================================"
echo "Starting baseline experiments!"
echo "Model: $MODEL"
echo "Node size: $NODE_SIZE"
echo "Graph type: $GRAPH_TYPE"
echo "Number of datasets: $NUM_DATASETS"
echo "Output directory: $OUTPUT_DIR"
echo "GPU range: 0-3"
echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================"

# Record start time
START_TIME=$(date +%s)

# Function to assign task to GPU
assign_baseline_task() {
    local gpu_id=$1
    local dataset_idx=$2
    local model=$3
    local node_size=$4
    local graph_type=$5
    
    local session_name="baseline_${model}_gpu${gpu_id}_${node_size}_${graph_type}_${dataset_idx}"
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Preparing GPU $gpu_id: $model dataset $dataset_idx"
    
    tmux kill-session -t "$session_name" 2>/dev/null || true
    tmux new-session -d -s "$session_name" -c "$WORK_DIR"
    tmux send-keys -t "$session_name" "conda activate $CONDA_ENV" Enter
    tmux send-keys -t "$session_name" "cd $WORK_DIR" Enter
    
    # 시작 신호 파일을 기다리게 함
    tmux send-keys -t "$session_name" "
        echo 'GPU $gpu_id: Model $model dataset $dataset_idx ready, waiting for start signal...'
        while [ ! -f /tmp/baseline_start_signal ]; do sleep 0.01; done
        echo 'GPU $gpu_id: Starting $model dataset $dataset_idx at \$(date +\"%Y-%m-%d %H:%M:%S\")'
        CUDA_VISIBLE_DEVICES=$gpu_id python experiment_script.py \\
            --model $model \\
            --exp_type synthetic \\
            --nodes $node_size \\
            --config_type $graph_type \\
            --dataset_indices \"${dataset_idx}-${dataset_idx}\" \\
            --output_dir $OUTPUT_DIR \\
            --gpu $gpu_id
        echo 'GPU $gpu_id: Model $model dataset $dataset_idx completed at \$(date +\"%Y-%m-%d %H:%M:%S\")'
        exit
    " Enter
    
    return 0
}

METHOD_DIR="$OUTPUT_DIR/nodes_${NODE_SIZE}/${GRAPH_TYPE}/${MODEL}"
if [ -d "$METHOD_DIR" ]; then
    echo "Removing existing results directory: $METHOD_DIR"
    rm -rf "$METHOD_DIR"
fi

rm -f /tmp/baseline_start_signal

GPUS=(0 1 2 3)
NUM_GPUS=${#GPUS[@]}

echo ""
echo "=== Preparing all $NUM_DATASETS tasks ==="
for dataset_idx in $(seq 1 $((NUM_DATASETS))); do
    # 블록 단위 GPU 할당 (0-based index)
    # GPU 0: 1-7 (7개), GPU 1: 8-15 (8개), GPU 2: 16-22 (7개), GPU 3: 23-29 (8개)
    if [ $dataset_idx -le 7 ]; then
        gpu_id=0
    elif [ $dataset_idx -le 15 ]; then
        gpu_id=1
    elif [ $dataset_idx -le 22 ]; then
        gpu_id=2
    else
        gpu_id=3
    fi
    
    assign_baseline_task $gpu_id $dataset_idx $MODEL $NODE_SIZE $GRAPH_TYPE &
    
    # 너무 빠른 실행 방지
    if [ $((dataset_idx % 10)) -eq 9 ]; then
        sleep 0.1
    fi
done

wait
echo "All tasks prepared!"

# 동시 시작!
echo ""
echo "================================================"
echo "🚀 STARTING ALL $NUM_DATASETS EXPERIMENTS SIMULTANEOUSLY!"
echo "Model: $MODEL"
echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================"
EXPERIMENT_START_TIME=$(date +%s)

touch /tmp/baseline_start_signal

# 백그라운드에서 진행 상황 모니터링 (선택사항)
echo ""
echo "Monitoring progress... (Press Ctrl+C to stop monitoring, experiments will continue)"
echo ""

monitor_progress() {
    while [ -f /tmp/baseline_start_signal ]; do
        sleep 10
        running=$(tmux ls 2>/dev/null | grep -c "baseline_${MODEL}" || echo "0")
        echo "[$(date '+%H:%M:%S')] Still running: $running sessions"
        
        # 모든 세션이 종료되면 루프 탈출
        if [ "$running" -eq 0 ]; then
            break
        fi
    done
}

# 사용자가 결과를 확인할 수 있도록 안내
echo "Experiments are running in background tmux sessions."
echo ""
echo "Useful commands:"
echo "  - List sessions:    tmux ls | grep baseline_${MODEL}"
echo "  - Attach to session: tmux attach -t baseline_${MODEL}_gpu0_${NODE_SIZE}_${GRAPH_TYPE}_1"
echo "  - Kill all sessions: tmux kill-session -t baseline_${MODEL}_gpu{0..3}*"
echo ""
echo "Results will be saved to: $OUTPUT_DIR/nodes_${NODE_SIZE}/${GRAPH_TYPE}/${MODEL}/"
