#!/bin/bash

# Environment variables setup
export WANDB_API_KEY=
export HF_TOKEN=

# Basic configuration
NUM_GPUS=8
BASE_MODEL=output/fft_7b_if
CONFIG_FILE=configs/train_configs/llama2_7b_if_local_masktune.yaml

# Define the masked layers configurations to test
# Format: "name|layers" where layers are space-separated numbers
CONFIGS=(
    "0-3-layers|0 1 2 3"
)

# Function to run a single experiment
run_experiment() {
    local config_name=$1
    local masked_layers=$2

    # Create experiment ID
    local EXP_ID="masktune_7b_local_${config_name}_0.9_if"
    local OUTPUT_DIR="output/$EXP_ID"
    local LOG_FILE="output/$EXP_ID/${EXP_ID}_$(date '+%Y%m%d_%H%M%S').log"

    echo "Starting experiment: $EXP_ID"
    echo "Masked layers: $masked_layers"

    {
        echo "==== STARTING EXPERIMENT: $EXP_ID ===="
        echo "Log File: $LOG_FILE"
        echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "Masked Layers: $masked_layers"
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
            --model_name_or_path=$BASE_MODEL \
            --output_dir=$OUTPUT_DIR \
            --masked_layers="$masked_layers"

        # Apply masks
        python scripts/apply_masks.py \
            --base_model_name_or_path $BASE_MODEL \
            --output_dir $OUTPUT_DIR/mask_applied \
            --mask_model_name_or_path $OUTPUT_DIR \
            --save_masks

        echo "==== EXPERIMENT COMPLETED: $EXP_ID ===="
        echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "====================================="
    } 2>&1 | tee "$LOG_FILE"
}

# Run all configurations
for config in "${CONFIGS[@]}"; do
    IFS='|' read -r config_name masked_layers <<< "$config"
    run_experiment "$config_name" "$masked_layers"
done

echo "All experiments completed!"