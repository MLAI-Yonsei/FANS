#!/bin/bash

# ===============================
# Single Batch Experiment Runner (GPU 0-3, ER only)
# ===============================

set -e

# Parse arguments
NODE_SIZE="${1:-nodes_10}"
GRAPH_TYPE="${2:-ER}"
CONFIG_DIR="${3:-data}"

# Common settings
CONDA_ENV="fans"
WORK_DIR="./fans"
PROJECT_NAME="causal_nf"

echo "================================================"
echo "Starting single batch experiments!"
echo "Config dir: $CONFIG_DIR"
echo "Node size: $NODE_SIZE"
echo "Graph type: $GRAPH_TYPE"
echo "GPU range: 0-3 (30 experiments total)"
echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================"

# Record start time
START_TIME=$(date +%s)

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
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Preparing GPU $gpu_id: $exp_name"
    
    local wandb_group=$(get_wandb_group "$config_file")
    local safe_config_dir="${CONFIG_DIR//\//_}"
    safe_config_dir="${safe_config_dir//./_}"
    local session_name="gpu${gpu_id}_${safe_config_dir}_${exp_name}"
    
    tmux kill-session -t "$session_name" 2>/dev/null || true
    tmux new-session -d -s "$session_name" -c "$WORK_DIR"
    tmux send-keys -t "$session_name" "conda activate $CONDA_ENV" Enter
    tmux send-keys -t "$session_name" "cd $WORK_DIR" Enter
    
    # 시작 신호 파일을 기다리게 함
    tmux send-keys -t "$session_name" "
        echo 'GPU $gpu_id: Task $exp_name ready, waiting for start signal...'
        while [ ! -f /tmp/fans_start_signal ]; do sleep 0.1; done
        echo 'GPU $gpu_id: Starting experiment $exp_name at \$(date +\"%Y-%m-%d %H:%M:%S\")'
        CUDA_VISIBLE_DEVICES=$gpu_id python main.py \\
            --config_file causal_nf/configs/${CONFIG_DIR}/$config_file \\
            --wandb_mode online \\
            --project $PROJECT_NAME \\
            --wandb_group '$wandb_group'
        echo 'GPU $gpu_id: Experiment $exp_name completed at \$(date +\"%Y-%m-%d %H:%M:%S\")'
        exit
    " Enter
    
    return 0
}

# 시작 신호 파일 삭제 (이전 실행 제거)
rm -f /tmp/fans_start_signal

# 모든 태스크 준비
echo "=== Preparing all 30 tasks ==="
for i in {1..30}; do
    if [ $i -le 8 ]; then
        gpu_id=0
    elif [ $i -le 15 ]; then
        gpu_id=1
    elif [ $i -le 23 ]; then
        gpu_id=2
    else
        gpu_id=3
    fi
    
    config_file="${NODE_SIZE}/${GRAPH_TYPE}/causal_nf_${NODE_SIZE}_${GRAPH_TYPE}_adj_${i}.yaml"
    exp_name="${NODE_SIZE}_${GRAPH_TYPE}_${i}"
    
    assign_task_to_gpu $gpu_id "$config_file" "$exp_name" &
done

wait
echo "All tasks prepared!"

# 준비 시간 확보 (conda activate 완료 대기)
echo "Waiting for all sessions to be ready (5 seconds)..."
sleep 5

# 동시 시작!
echo ""
echo "================================================"
echo "🚀 STARTING ALL 30 EXPERIMENTS SIMULTANEOUSLY!"
echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================"
START_TIME=$(date +%s)

touch /tmp/fans_start_signal
