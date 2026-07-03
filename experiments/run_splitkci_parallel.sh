#!/bin/bash

# ===============================
# SplitKCI Multi-Node Runner (All configurations)
# ===============================

set -e

# Common settings
CONDA_ENV="fans"
WORK_DIR="/home/statduck/fans/experiments"
OUTPUT_DIR="$WORK_DIR/costs"
NUM_DATASETS=30

# Slack Webhook URL (여기에 직접 입력)
SLACK_WEBHOOK_URL="https://hooks.slack.com/services/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"

# 실험 설정
NODE_SIZES=(10 20 30 40 50)
GRAPH_TYPES=("ER" "SF")

TOTAL_EXPERIMENTS=$((${#NODE_SIZES[@]} * ${#GRAPH_TYPES[@]}))

echo "================================================"
echo "SplitKCI Multi-Node Execution"
echo "Node sizes: ${NODE_SIZES[@]}"
echo "Graph types: ${GRAPH_TYPES[@]}"
echo "Total experiments: $TOTAL_EXPERIMENTS"
echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================"

# Slack 메시지 전송 함수
send_slack_message() {
    local message=$1
    local color=${2:-"good"}
    
    if [ -z "$SLACK_WEBHOOK_URL" ]; then
        echo "[Slack] Webhook URL not set, skipping"
        return
    fi
    
    curl -s -X POST "$SLACK_WEBHOOK_URL" \
        -H "Content-Type: application/json" \
        -d "{
            \"attachments\": [{
                \"color\": \"$color\",
                \"text\": \"$message\",
                \"footer\": \"SplitKCI Runner\",
                \"ts\": $(date +%s)
            }]
        }" > /dev/null
    
    echo "[Slack] Message sent"
}

# ========== 메인 실행 ==========

EXPERIMENT_NUM=0

for NODE_SIZE in "${NODE_SIZES[@]}"; do
    for GRAPH_TYPE in "${GRAPH_TYPES[@]}"; do
        EXPERIMENT_NUM=$((EXPERIMENT_NUM + 1))
        
        echo ""
        echo "========================================================"
        echo "[$EXPERIMENT_NUM/$TOTAL_EXPERIMENTS] nodes_${NODE_SIZE} / ${GRAPH_TYPE}"
        echo "Start: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "========================================================"
        
        EXP_START_TIME=$(date +%s)
        
        # SplitKCI 직접 실행
        cd "$WORK_DIR"
        
        python experiment_script.py \
            --model splitkci \
            --exp_type synthetic \
            --nodes ${NODE_SIZE} \
            --config_type ${GRAPH_TYPE} \
            --dataset_indices "1-${NUM_DATASETS}" \
            --output_dir ${OUTPUT_DIR} \
            --gpu -1
        
        EXP_END_TIME=$(date +%s)
        EXP_DURATION_MIN=$(( (EXP_END_TIME - EXP_START_TIME) / 60 ))
        
        # 결과 확인
        RESULTS_DIR="$OUTPUT_DIR/nodes_${NODE_SIZE}/${GRAPH_TYPE}/splitkci"
        if [ -d "$RESULTS_DIR" ]; then
            total_results=$(ls "$RESULTS_DIR"/*.json 2>/dev/null | wc -l)
        else
            total_results=0
        fi
        
        # Slack 알림
        if [ "$total_results" -ge "$NUM_DATASETS" ]; then
            send_slack_message "✅ [$EXPERIMENT_NUM/$TOTAL_EXPERIMENTS] *SplitKCI* 완료\n• Nodes: ${NODE_SIZE}, Graph: ${GRAPH_TYPE}\n• Results: ${total_results}/${NUM_DATASETS}\n• Duration: ${EXP_DURATION_MIN}분" "good"
        else
            send_slack_message "⚠️ [$EXPERIMENT_NUM/$TOTAL_EXPERIMENTS] *SplitKCI* 완료 (일부 누락)\n• Nodes: ${NODE_SIZE}, Graph: ${GRAPH_TYPE}\n• Results: ${total_results}/${NUM_DATASETS}\n• Duration: ${EXP_DURATION_MIN}분" "warning"
        fi
        
        echo "[$EXPERIMENT_NUM/$TOTAL_EXPERIMENTS] Completed: ${total_results}/${NUM_DATASETS} (${EXP_DURATION_MIN}min)"
    done
done

# 전체 완료 알림
send_slack_message "🎉 *SplitKCI 모든 실험 완료!*\n• Node sizes: 10, 20, 30, 40, 50\n• Graph types: ER, SF\n• Total: ${TOTAL_EXPERIMENTS} experiments\n• End: $(date '+%Y-%m-%d %H:%M:%S')" "good"

echo ""
echo "================================================"
echo "🎉 All experiments completed!"
echo "End time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================"
