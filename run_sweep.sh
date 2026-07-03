#!/bin/bash

# ===============================
# Hyperparameter Sweep Runner
# Runs 1 sweep folder at a time across 4 GPUs (8/7/8/7 split)
# ===============================

set -e

# Repo root = directory containing this script (works when run from fans/ or elsewhere)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CONDA_ENV="fans"
WORK_DIR="$SCRIPT_DIR"
PROJECT_NAME="causal_nf"
SWEEP_BASE="data_sweep"
NODE_SIZE="nodes_50"
GRAPH_TYPE="ER"

# Which sweep folders to run: nsf | naf | all
# Override when invoking, e.g.:  SWEEP_FAMILY=all ./run_sweep.sh
SWEEP_FAMILY="${SWEEP_FAMILY:-all}"

ALL_SWEEP_DIRS=(
    "naf_bs512_lr001"
    "naf_bs1024_lr001"
    "naf_bs2048_lr001"
    "naf_bs4096_lr001"
    "naf_bs1024_lr0001"
    "naf_bs512_lr0001"
    "naf_bs2048_lr0001"
    "naf_bs4096_lr0001"
    "nsf_bs512_lr001"
    "nsf_bs1024_lr001"
    "nsf_bs2048_lr001"
    "nsf_bs4096_lr001"
    "nsf_bs512_lr0001"
    "nsf_bs1024_lr0001"
    "nsf_bs2048_lr0001"
    "nsf_bs4096_lr0001"
)

SWEEP_DIRS=()
for d in "${ALL_SWEEP_DIRS[@]}"; do
    case "$SWEEP_FAMILY" in
        all) SWEEP_DIRS+=("$d") ;;
        nsf) [[ "$d" == nsf_* ]] && SWEEP_DIRS+=("$d") ;;
        naf) [[ "$d" == naf_* ]] && SWEEP_DIRS+=("$d") ;;
        *)
            echo "Invalid SWEEP_FAMILY='$SWEEP_FAMILY' (use nsf, naf, or all)" >&2
            exit 1
            ;;
    esac
done

if [ "${#SWEEP_DIRS[@]}" -eq 0 ]; then
    echo "No sweep dirs for SWEEP_FAMILY=$SWEEP_FAMILY" >&2
    exit 1
fi

echo "================================================"
echo "Hyperparameter Sweep Runner"
echo "SWEEP_FAMILY=$SWEEP_FAMILY  |  Total settings: ${#SWEEP_DIRS[@]}"
echo "Running 1 folder at a time on 4 GPUs (8/7/8/7 split)"
echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================"

echo "================================================"

launch_folder() {
    local sweep_dir=$1
    local gpu_a=$2
    local gpu_b=$3
    local gpu_c=$4
    local gpu_d=$5
    local signal_file=$6

    local config_dir="${SWEEP_BASE}/${sweep_dir}"
    local safe_name="${sweep_dir//\//_}"

    echo "  Launching $sweep_dir on GPU $gpu_a,$gpu_b,$gpu_c,$gpu_d (8/7/8/7 split)"

    for i in $(seq 1 30); do
        # Distribute 30 configs across 4 GPUs as 8/7/8/7 to keep per-GPU
        # memory pressure low enough to avoid OOM with large batch sizes.
        if [ $i -le 8 ]; then
            gpu_id=$gpu_a
        elif [ $i -le 15 ]; then
            gpu_id=$gpu_b
        elif [ $i -le 23 ]; then
            gpu_id=$gpu_c
        else
            gpu_id=$gpu_d
        fi

        local config_file="${NODE_SIZE}/${GRAPH_TYPE}/causal_nf_${NODE_SIZE}_${GRAPH_TYPE}_adj_${i}.yaml"
        local session_name="sweep_${safe_name}_gpu${gpu_id}_${i}"

        tmux kill-session -t "$session_name" 2>/dev/null || true
        tmux new-session -d -s "$session_name" -c "$WORK_DIR"

        tmux send-keys -t "$session_name" "conda activate $CONDA_ENV" Enter
        tmux send-keys -t "$session_name" "cd $WORK_DIR" Enter

        tmux send-keys -t "$session_name" "
            while [ ! -f $signal_file ]; do sleep 0.1; done
            CUDA_VISIBLE_DEVICES=$gpu_id python main.py \\
                --config_file causal_nf/configs/${config_dir}/$config_file \\
                --wandb_mode online \\
                --project $PROJECT_NAME \\
                --wandb_group '${config_dir}/${NODE_SIZE}/${GRAPH_TYPE}' || true
            exit
        " Enter
    done
}

wait_for_sweep_sessions() {
    echo "  Waiting for all sweep sessions to finish (polling every 5 min)..."
    # Locally relax 'set -e' so transient tmux/pgrep failures don't kill the
    # runner and leave orphan tmux sessions stuck in their signal-file wait.
    set +e
    while true; do
        local tmux_count
        tmux_count="$(tmux ls 2>/dev/null | awk -F: '/^sweep_/ {n++} END {printf "%d", n+0}')"
        [ -z "$tmux_count" ] && tmux_count=0

        # Count real python workers: a crashed python that left only a shell
        # prompt inside tmux should NOT block progress to the next round.
        local py_count
        py_count="$(pgrep -fc "python main.py --config_file causal_nf/configs/${SWEEP_BASE}/" 2>/dev/null)"
        [ -z "$py_count" ] && py_count=0

        if [ "$py_count" -eq 0 ]; then
            if [ "$tmux_count" -gt 0 ]; then
                echo "  [$(date '+%Y-%m-%d %H:%M:%S')] python=0 but $tmux_count tmux sweep_* sessions remain; killing them."
                tmux ls 2>/dev/null | awk -F: '/^sweep_/ {print $1}' \
                    | xargs -r -n1 -I{} tmux kill-session -t {} >/dev/null 2>&1
            fi
            break
        fi
        echo "  [$(date '+%Y-%m-%d %H:%M:%S')] Running: python=$py_count, tmux=$tmux_count"
        sleep 300
    done
    set -e
    echo "  All sessions completed."
}

total=${#SWEEP_DIRS[@]}
round=1

for ((idx=0; idx<total; idx++)); do
    dir_a="${SWEEP_DIRS[$idx]}"

    signal_file="/tmp/sweep_signal_round${round}"
    rm -f "$signal_file"

    echo ""
    echo "================================================"
    echo "Round $round: $dir_a"
    echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "================================================"

    launch_folder "$dir_a" 0 1 2 3 "$signal_file"

    sleep 5
    touch "$signal_file"
    echo "  Started round $round!"

    wait_for_sweep_sessions
    rm -f "$signal_file"

    echo "Round $round completed at $(date '+%Y-%m-%d %H:%M:%S')"
    round=$((round + 1))
done

echo ""
echo "================================================"
echo "All sweep experiments completed!"
echo "End time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================"
