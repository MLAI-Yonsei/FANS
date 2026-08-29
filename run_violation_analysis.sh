#!/bin/bash

# =========================================================================
# Violation ablation: re-run the FANS analysis only (no training).
#
# The NSF/512 checkpoints under /data1 are complete, but they were produced
# by an older code path that ran with simultaneous_shift=False, so
# fans_results.json has simultaneous_shift_results = null. Passing
# --load_model reuses the best checkpoint and only re-runs detection plus
# the simultaneous-shift dissection, overwriting fans_results.json in place.
#
# Usage: bash run_violation_analysis.sh <lambda> [start_idx] [end_idx]
#   bash run_violation_analysis.sh 1.00        # datasets 16-30
#   bash run_violation_analysis.sh 0.50 16 30
# =========================================================================

set -e

LAMBDA="${1:?lambda required, e.g. 0.50 or 1.00}"
START_IDX="${2:-16}"
END_IDX="${3:-30}"

NODE_SIZE="nodes_10"
GRAPH_TYPE="ER"
CONDA_ENV="fans"
WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULT_ROOT="/data1/statduck/exp1_result_fans_violation_lambda_${LAMBDA}"
SIGNAL_FILE="/tmp/fans_violation_signal_${LAMBDA}"
NUM_GPUS=4

echo "================================================"
echo "Violation analysis re-run (analysis only)"
echo "lambda      : $LAMBDA"
echo "datasets    : $START_IDX-$END_IDX"
echo "work dir    : $WORK_DIR"
echo "result root : $RESULT_ROOT"
echo "================================================"

# ---- Pre-flight: resolve every config and checkpoint before touching tmux ----
declare -A CONFIGS RUN_DIRS

for i in $(seq "$START_IDX" "$END_IDX"); do
    config="causal_nf/configs/data_violation/lambda_${LAMBDA}/${NODE_SIZE}/${GRAPH_TYPE}/causal_nf_${NODE_SIZE}_${GRAPH_TYPE}_adj_${i}.yaml"
    if [ ! -f "$WORK_DIR/$config" ]; then
        echo "ERROR: config not found: $WORK_DIR/$config" >&2
        exit 1
    fi

    parent="${RESULT_ROOT}/data1_statduck_data_both_data_violation_lambda_${LAMBDA}_${NODE_SIZE}_${GRAPH_TYPE}_adj_${i}"
    ckpt=$(ls "$parent"/*/epoch=*.ckpt 2>/dev/null | head -1 || true)
    if [ -z "$ckpt" ]; then
        echo "ERROR: no epoch=*.ckpt under $parent" >&2
        exit 1
    fi

    CONFIGS[$i]="$config"
    RUN_DIRS[$i]="$(dirname "$ckpt")"
    echo "  adj_${i}: $(basename "${RUN_DIRS[$i]}")  <-  $(basename "$ckpt")"
done

echo ""
echo "All $((END_IDX - START_IDX + 1)) checkpoints resolved. Launching tmux sessions."
echo ""

rm -f "$SIGNAL_FILE"

launch_task() {
    local gpu_id=$1
    local idx=$2
    local config=${CONFIGS[$idx]}
    local run_dir=${RUN_DIRS[$idx]}
    local session_name="gpu${gpu_id}_viol${LAMBDA//./_}_adj${idx}"

    tmux kill-session -t "$session_name" 2>/dev/null || true
    tmux new-session -d -s "$session_name" -c "$WORK_DIR"
    tmux send-keys -t "$session_name" "conda activate $CONDA_ENV" Enter
    tmux send-keys -t "$session_name" "cd $WORK_DIR" Enter
    tmux send-keys -t "$session_name" "
        echo 'GPU $gpu_id: adj_${idx} waiting for start signal...'
        while [ ! -f $SIGNAL_FILE ]; do sleep 0.1; done
        echo 'GPU $gpu_id: adj_${idx} started at \$(date +\"%Y-%m-%d %H:%M:%S\")'
        CUDA_VISIBLE_DEVICES=$gpu_id python main.py \\
            --config_file $config \\
            --load_model $run_dir
        echo 'GPU $gpu_id: adj_${idx} finished at \$(date +\"%Y-%m-%d %H:%M:%S\")'
        exit
    " Enter
}

task_no=0
for i in $(seq "$START_IDX" "$END_IDX"); do
    gpu_id=$((task_no % NUM_GPUS))
    echo "[$(date '+%H:%M:%S')] GPU $gpu_id <- adj_${i}"
    launch_task "$gpu_id" "$i"
    task_no=$((task_no + 1))
done

echo ""
echo "Waiting 5 seconds for conda activation..."
sleep 5

echo "================================================"
echo "Starting $task_no analysis runs at $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================"
touch "$SIGNAL_FILE"

echo ""
echo "Monitor with : tmux ls"
echo "Attach with  : tmux attach -t gpu0_viol${LAMBDA//./_}_adj${START_IDX}"
