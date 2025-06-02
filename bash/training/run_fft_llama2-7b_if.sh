#!/bin/bash

# Environment variables setup
export WANDB_API_KEY=
export HF_TOKEN=

# Basic configuration
NUM_GPUS=8
CONFIG_FILE=configs/train_configs/llama2_7b_if_fft.yaml
EXP_ID=fft_7b_if
OUTPUT_DIR=output/$EXP_ID
LOG_FILE=output/$EXP_ID/${EXP_ID}_$(date '+%Y%m%d_%H%M%S').log

echo "Number of GPUs: $NUM_GPUS"
echo "Using config file: $CONFIG_FILE"

{
    echo "==== STARTING EXPERIMENT: $EXP_ID ===="
    echo "Log File: $LOG_FILE"
    echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "====================================="

    # You can also set --gradient_checkpointing or use `stage3_offloading_accelerate.conf` to save memory,
    # but it will trade off speed.
    accelerate launch \
        --main_process_port 29400 \
        --mixed_precision bf16 \
        --num_machines 1 \
        --num_processes $NUM_GPUS \
        --use_deepspeed \
        --deepspeed_config_file configs/ds_configs/stage3_no_offloading_accelerate.conf \
        scripts/finetune.py \
        "$CONFIG_FILE" \
        --output_dir=$OUTPUT_DIR

    echo "==== EXPERIMENT COMPLETED: $EXP_ID ===="
    echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "====================================="
} 2>&1 | tee "$LOG_FILE"