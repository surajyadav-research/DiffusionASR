#!/bin/bash

# Speech-Align-LLM Manifest Creation Script
# Usage: ./create_manifests.sh [config_file]
# Example: ./create_manifests.sh config_wavlm_linear_llama.yaml

set -e  # Exit on any error

# Default config file
CONFIG_FILE=${1:-"/home/puneets/speech-align-llm/configs/hindi/lora_openai-whisper-large-v3_multi-res-conv-large-projector_krutrim-2-12b-instruct.yaml"}

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file '$CONFIG_FILE' not found!"
    echo "Usage: $0 [config_file]"
    echo "Available config files:"
    ls -1 config_*.yaml 2>/dev/null || echo "  No config files found"
    exit 1
fi

echo "=========================================="
echo "Speech-Align-LLM Manifest Creation"
echo "=========================================="
echo "Config file: $CONFIG_FILE"
echo "Start time: $(date)"
echo "=========================================="

# Get Python executable path (assuming conda environment)
PYTHON_PATH="/home/abrol/miniconda/envs/suksham/bin/python"

# Check if Python executable exists
if [ ! -f "$PYTHON_PATH" ]; then
    echo "Warning: Python path $PYTHON_PATH not found. Using system python."
    PYTHON_PATH="python"
fi

# Create manifest files
echo ""
echo "Creating manifest files..."
echo "----------------------------------------"
$PYTHON_PATH create_manifest_files.py --config "$CONFIG_FILE"

if [ $? -eq 0 ]; then
    echo "[SUCCESS] Manifest files created successfully"
else
    echo "[ERROR] Failed to create manifest files"
    exit 1
fi

echo ""
echo "=========================================="
echo "Manifest Creation Summary"
echo "=========================================="
echo "Config file: $CONFIG_FILE"
echo "End time: $(date)"
echo "=========================================="

echo "[SUCCESS] Manifest creation completed successfully!" 