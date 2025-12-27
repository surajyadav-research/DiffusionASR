#!/bin/bash

# Simple Speech-Align-LLM Training Script
# Usage: ./train.sh [config_file]
# Example: ./train.sh config_wavlm_linear_llama.yaml

set -e  # Exit on any error

# Default config file
CONFIG_FILE=${1}

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file '$CONFIG_FILE' not found!"
    echo "Usage: $0 [config_file]"
    echo "Available config files:"
    ls -1 config_*.yaml 2>/dev/null || echo "  No config files found"
    exit 1
fi

echo "=========================================="
echo "Speech-Align-LLM Training"
echo "=========================================="
echo "  Config file: $CONFIG_FILE"
echo "  Start time: $(date)"
echo "=========================================="

# Get Python executable path (assuming conda environment)
PYTHON_PATH="/home/abrol/miniconda/envs/suksham/bin/python"

# Check if Python executable exists
if [ ! -f "$PYTHON_PATH" ]; then
    echo "  Warning: Python path $PYTHON_PATH not found. Using system python."
    PYTHON_PATH="python"
fi

# Train the model
echo ""
echo "Training the model..."
echo "----------------------------------------"
echo "  Executing: train.py --path_to_config_file $CONFIG_FILE"
$PYTHON_PATH train.py --path_to_config_file "$CONFIG_FILE"

if [ $? -eq 0 ]; then
    echo "  [SUCCESS] Model training completed successfully"
    echo "  End time: $(date)"
    echo "=========================================="
else
    echo "  [ERROR] Model training failed"
    exit 1
fi
