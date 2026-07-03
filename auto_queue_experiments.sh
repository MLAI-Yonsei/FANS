#!/bin/bash

# ===============================
# Auto Queue Experiment Monitor
# ===============================

set -e

# Settings
WORK_DIR="/home/statduck/fans"
GPU_IDS=(0 1 2 3)
MEMORY_THRESHOLD=400  # MB
CHECK_INTERVAL=300    # seconds (5분마다 체크)
MAX_WAIT_TIME=86400   # 24 hours
SLACK_WEBHOOK_URL="${SLACK_WEBHOOK_URL:-}"

# === Slack notification (uses $SLACK_WEBHOOK_URL env var) ===
send_slack() {
    local msg="$1"
    if [ -z "${SLACK_WEBHOOK_URL:-}" ]; then
        echo "[slack] SLACK_WEBHOOK_URL not set; skipping notification: $msg"
        return 0
    fi
    local escaped="${msg//\\/\\\\}"
    escaped="${escaped//\"/\\\"}"
    escaped="${escaped//$'\n'/\\n}"
    curl -s -X POST -H 'Content-type: application/json' \
        --data "{\"text\":\"${escaped}\"}" "$SLACK_WEBHOOK_URL" > /dev/null || true
}

format_elapsed() {
    local s=$1
    printf '%dh %02dm %02ds' $((s/3600)) $(((s%3600)/60)) $((s%60))
}

# Define experiment queue (node_size:graph_type:config_dir)
# config_dir defaults to "data" if omitted (3rd field).
# data_anm + data_swap + data_violation:
#   data_anm        × {nodes_10, nodes_50} × ER = 2 batches
#   data_swap       × {nodes_10, nodes_50} × ER = 2 batches
#   data_violation  × {lambda_0.50, lambda_1.00} × {nodes_10, nodes_50} × ER = 4 batches
#   Total: 8 batches × 30 exps = 240 experiments
# data_sweep/naf_bs1024_lr001:
#   {nodes_10, nodes_50} × ER = 2 batches × 30 exps = 60 experiments
declare -a EXPERIMENT_QUEUE=(
    "nodes_50:ER:data_anm"
)

echo "================================================"
echo "Auto Queue Experiment Monitor"
echo "Monitoring GPUs: ${GPU_IDS[@]}"
echo "Memory threshold: ${MEMORY_THRESHOLD}MB"
echo "Check interval: ${CHECK_INTERVAL}s"
echo "Queued experiments: ${#EXPERIMENT_QUEUE[@]}"
echo "================================================"

for exp in "${EXPERIMENT_QUEUE[@]}"; do
    IFS=':' read -r node_size graph_type config_dir <<< "$exp"
    config_dir="${config_dir:-data}"
    echo "  - ${config_dir}/${node_size}/${graph_type}"
done
echo ""

# Function to check GPU memory usage
check_gpu_memory() {
    local gpu_id=$1
    local mem_used=$(nvidia-smi --id=$gpu_id --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null || echo "0")
    echo "$mem_used"
}

# Function to check if all GPUs are idle
all_gpus_idle() {
    for gpu_id in "${GPU_IDS[@]}"; do
        local mem_used=$(check_gpu_memory $gpu_id)
        
        if [ $(echo "$mem_used > $MEMORY_THRESHOLD" | bc) -eq 1 ]; then
            return 1  # Not idle
        fi
    done
    return 0  # All idle
}

# Function to check running tmux sessions
check_tmux_sessions() {
    local session_count=$(tmux ls 2>/dev/null | grep -E "gpu[0-3]_" | wc -l || echo "0")
    echo "$session_count"
}

# Function to wait for GPUs to be idle
wait_for_idle() {
    local wait_time=0
    
    echo ""
    echo "Waiting for all GPUs to become idle..."
    
    while true; do
        session_count=$(check_tmux_sessions)
        
        if all_gpus_idle && [ "$session_count" -eq 0 ]; then
            echo ""
            echo "✓ All GPUs are idle!"
            return 0
        fi
        
        if [ $wait_time -ge $MAX_WAIT_TIME ]; then
            echo ""
            echo "⚠ WARNING: Maximum wait time (${MAX_WAIT_TIME}s) reached"
            echo "Skipping to next experiment..."
            return 1
        fi
        
        # Status update every 5 checks
        if [ $((wait_time % (CHECK_INTERVAL * 5))) -eq 0 ]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Waiting... (${wait_time}s elapsed)"
            for gpu_id in "${GPU_IDS[@]}"; do
                local mem_used=$(check_gpu_memory $gpu_id)
                echo "  GPU $gpu_id: ${mem_used}MB"
            done
            echo "  Active sessions: $session_count"
        fi
        
        sleep "$CHECK_INTERVAL"
        wait_time=$((wait_time + CHECK_INTERVAL))
    done
}

# Main execution loop
echo "================================================"
echo "Starting experiment queue execution"
echo "================================================"

PREV_START_TIME=""
PREV_LABEL=""

for exp_index in "${!EXPERIMENT_QUEUE[@]}"; do
    exp="${EXPERIMENT_QUEUE[$exp_index]}"
    IFS=':' read -r node_size graph_type config_dir <<< "$exp"
    config_dir="${config_dir:-data}"
    
    exp_num=$((exp_index + 1))
    total_exp=${#EXPERIMENT_QUEUE[@]}
    current_label="${config_dir}/${node_size}/${graph_type}"
    
    echo ""
    echo "================================================"
    echo "Experiment ${exp_num}/${total_exp}: ${current_label}"
    echo "================================================"
    
    # Wait for GPUs to be idle (skip for first experiment)
    if [ $exp_index -gt 0 ]; then
        if wait_for_idle; then
            prev_elapsed=$(( $(date +%s) - PREV_START_TIME ))
            send_slack "[$((exp_num-1))/${total_exp}] ${PREV_LABEL} 30개 완료 ($(format_elapsed $prev_elapsed))"
        fi
        
        # Additional safety wait
        echo "Waiting 30 seconds before starting next experiment..."
        sleep 30
    fi
    
    # Run experiment
    cd "$WORK_DIR"
    
    PREV_START_TIME=$(date +%s)
    PREV_LABEL="${current_label}"
    
    echo "Starting experiment: ${current_label}"
    bash run_experiments.sh "$node_size" "$graph_type" "$config_dir"
    
    echo ""
    echo "Experiment ${exp_num}/${total_exp} launched successfully!"
    
    # Brief wait to ensure tmux sessions are created
    sleep 10
done

# Wait for the LAST batch to finish (loop doesn't wait for it)
echo ""
echo "================================================"
echo "Waiting for the last batch (${PREV_LABEL}) to finish..."
echo "================================================"
if wait_for_idle; then
    last_elapsed=$(( $(date +%s) - PREV_START_TIME ))
    send_slack "[${#EXPERIMENT_QUEUE[@]}/${#EXPERIMENT_QUEUE[@]}] ${PREV_LABEL} 30개 완료 ($(format_elapsed $last_elapsed))"
fi

echo ""
echo "================================================"
echo "All experiments in queue have been launched!"
echo "================================================"
echo ""
echo "Monitor commands:"
echo "  tmux ls              # List active sessions"
echo "  nvidia-smi           # Check GPU usage"
echo "  tmux attach -t <name> # Attach to session"