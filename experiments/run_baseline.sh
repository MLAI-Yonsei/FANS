#!/bin/bash

# ===============================
# Baseline Model Batch Runner with GPU Memory Monitoring
# ===============================

set -e

# Parse arguments
MODEL="${1:-splitkci}"
NODE_SIZE="${2:-10}"
GRAPH_TYPE="${3:-ER}"
NUM_DATASETS="${4:-30}"

# Common settings
CONDA_ENV="fans"
WORK_DIR="/home/statduck/fans/experiments"
OUTPUT_DIR="$WORK_DIR/results/$MODEL"  # 결과 저장 경로: results/{model}/nodes_{size}/{graph_type}/
GPUS=(0 1 2 3)

echo "================================================"
echo "Batch-based baseline experiments!"
echo "Model: $MODEL"
echo "Node size: $NODE_SIZE"
echo "Graph type: $GRAPH_TYPE"
echo "Number of datasets: $NUM_DATASETS"
echo "Output directory: $OUTPUT_DIR"
echo "GPUs: ${GPUS[@]}"
echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================"

# GPU 메모리 확인 함수 (nvidia-smi 사용)
check_gpu_memory() {
    local gpu_id=$1
    # GPU 메모리 사용량 (MB 단위)
    local mem_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $gpu_id)
    echo $mem_used
}

# 모든 GPU 메모리가 임계값 이하인지 확인
all_gpus_free() {
    local threshold=${1:-100}  # 기본 100MB
    
    for gpu_id in "${GPUS[@]}"; do
        local mem=$(check_gpu_memory $gpu_id)
        if [ "$mem" -gt "$threshold" ]; then
            return 1  # 아직 메모리 사용 중
        fi
    done
    return 0  # 모든 GPU 여유
}

# 배치 실행 함수
run_batch() {
    local batch_num=$1
    shift
    local assignments=("$@")  # "gpu_id:start-end" 형식
    
    echo ""
    echo "================================================"
    echo "Starting Batch $batch_num"
    echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "================================================"
    
    # 신호 파일 삭제
    rm -f /tmp/baseline_start_signal_batch${batch_num}
    
    # 각 GPU별 태스크 생성
    for assignment in "${assignments[@]}"; do
        IFS=':' read -r gpu_id range <<< "$assignment"
        IFS='-' read -r start_idx end_idx <<< "$range"
        
        echo "GPU $gpu_id: datasets $start_idx-$end_idx"
        
        for dataset_idx in $(seq $start_idx $end_idx); do
            local session_name="baseline_${MODEL}_gpu${gpu_id}_${NODE_SIZE}_${GRAPH_TYPE}_${dataset_idx}"
            
            tmux kill-session -t "$session_name" 2>/dev/null || true
            tmux new-session -d -s "$session_name"
            tmux send-keys -t "$session_name" "cd $WORK_DIR" Enter
            tmux send-keys -t "$session_name" "conda activate $CONDA_ENV" Enter
            
            tmux send-keys -t "$session_name" "
                echo 'Batch $batch_num - GPU $gpu_id: Dataset $dataset_idx ready'
                while [ ! -f /tmp/baseline_start_signal_batch${batch_num} ]; do sleep 0.01; done
                echo 'Batch $batch_num - GPU $gpu_id: Starting dataset $dataset_idx at \$(date +\"%H:%M:%S\")'
                CUDA_VISIBLE_DEVICES=$gpu_id python experiment_script.py \\
                    --model $MODEL \\
                    --exp_type synthetic \\
                    --nodes $NODE_SIZE \\
                    --config_type $GRAPH_TYPE \\
                    --dataset_indices \"${dataset_idx}-${dataset_idx}\" \\
                    --output_dir $OUTPUT_DIR \\
                    --gpu $gpu_id
                echo 'Batch $batch_num - GPU $gpu_id: Dataset $dataset_idx completed at \$(date +\"%H:%M:%S\")'
                exit
            " Enter
        done
    done
    
    # 동시 시작 신호
    touch /tmp/baseline_start_signal_batch${batch_num}
    echo "🚀 Batch $batch_num started!"
}

# 배치 완료 대기 함수
wait_for_batch_completion() {
    local batch_num=$1
    local memory_threshold=${2:-100}  # 100MB
    
    echo ""
    echo "Waiting for batch $batch_num to complete..."
    echo "Monitoring GPU memory (threshold: ${memory_threshold}MB)"
    
    while true; do
        # tmux 세션 개수 확인 (출력 정리)
        local count=$(tmux ls 2>/dev/null | grep -c "baseline_${MODEL}" 2>/dev/null)
        local running=${count:-0}  # 빈 값이면 0
        
        if [ "$running" -eq 0 ]; then
            echo "All sessions finished. Checking GPU memory..."
            
            # GPU 메모리 확인
            if all_gpus_free $memory_threshold; then
                echo "✓ All GPUs free (< ${memory_threshold}MB)"
                break
            else
                echo "Waiting for GPU memory to clear..."
                for gpu_id in "${GPUS[@]}"; do
                    local mem=$(check_gpu_memory $gpu_id)
                    echo "  GPU $gpu_id: ${mem}MB"
                done
            fi
        else
            echo "[$(date '+%H:%M:%S')] Running sessions: $running"
        fi
        
        sleep 1
    done
    
    # 신호 파일 정리
    rm -f /tmp/baseline_start_signal_batch${batch_num}
    
    echo "Batch $batch_num completed!"
}

# ========== 메인 실행 ==========

# Batch 1: GPU 0(1-3), GPU 1(4-6), GPU 2(7-9), GPU 3(10-12)
BATCH1=(
    "0:1-3"
    "1:4-6"
    "2:7-9"
    "3:10-12"
)

run_batch 1 "${BATCH1[@]}"
wait_for_batch_completion 1 100

echo ""
echo "================================================"
echo "Batch 1 완료. Batch 2 시작..."
echo "================================================"

# Batch 2: GPU 0(13-15), GPU 1(16-18), GPU 2(19-21), GPU 3(22-24)
BATCH2=(
    "0:13-15"
    "1:16-18"
    "2:19-21"
    "3:22-24"
)

run_batch 2 "${BATCH2[@]}"
wait_for_batch_completion 2 100

echo ""
echo "================================================"
echo "Batch 2 완료. Batch 3 시작..."
echo "================================================"

# Batch 3: GPU 0(25-27), GPU 1(28-30)
BATCH3=(
    "0:25-27"
    "1:28-30"
)

run_batch 3 "${BATCH3[@]}"
wait_for_batch_completion 3 100

echo ""
echo "================================================"
echo "🎉 All batches completed!"
echo "End time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================"

# 결과 요약
echo ""
echo "Checking results..."
RESULTS_DIR="$OUTPUT_DIR/nodes_${NODE_SIZE}/${GRAPH_TYPE}"
if [ -d "$RESULTS_DIR" ]; then
    cd "$RESULTS_DIR"
    total_results=$(ls ${MODEL}_nodes${NODE_SIZE}_${GRAPH_TYPE}_*.json 2>/dev/null | wc -l)
    echo "Results directory: $RESULTS_DIR"
    echo "Total results generated: $total_results / $NUM_DATASETS"
    
    if [ "$total_results" -lt "$NUM_DATASETS" ]; then
        echo ""
        echo "Missing datasets:"
        for i in $(seq 1 $NUM_DATASETS); do
            if [ ! -f "${MODEL}_nodes${NODE_SIZE}_${GRAPH_TYPE}_${i}.json" ]; then
                echo "  - Dataset $i"
            fi
        done
    fi
else
    echo "Results directory not found: $RESULTS_DIR"
fi
