#!/bin/bash

export HF_TOKEN=

MODEL=$1
MODEL_PATH=""
OUTPUT_PATH=""

if [[ "$MODEL" == *"masktune"* ]]; then
    MODEL_PATH="output/$MODEL/mask_applied"
    OUTPUT_PATH="output/$MODEL/mask_applied/eval_output"
elif [[ "$MODEL" == *"fft"* ]]; then
    MODEL_PATH="output/$MODEL"
    OUTPUT_PATH="output/$MODEL/eval_output"
else
    MODEL_PATH="$MODEL"
    OUTPUT_PATH="output/$MODEL/eval_output"
fi

# Create experiment ID
EXP_ID="Eval_${MODEL}.log"
LOG_FILE="${EXP_ID}_$(date '+%Y%m%d_%H%M%S').log"

echo "Starting experiment: $EXP_ID"
{
    echo "==== STARTING EXPERIMENT: $EXP_ID ===="
    echo "Log File: $LOG_FILE"
    echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "====================================="

    olmes --model "$MODEL_PATH" \
          --model-type "vllm" \
          --task '{"task_name": "codex_humaneval", "num_shots": 0}' \
                 '{"task_name": "codex_humanevalplus", "num_shots": 0}' \
          --output-dir "$OUTPUT_PATH"

    echo "==== EXPERIMENT COMPLETED: $EXP_ID ===="
    echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "====================================="
} 2>&1 | tee "$LOG_FILE"