#!/bin/bash

# Base directories
MUCS_DIR="/home/abrol/puneet/speech-align-llm/mucs"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/../mucs_create_manifest_files.py"

# Create output directories
mkdir -p "$MUCS_DIR/data/processed/train"
mkdir -p "$MUCS_DIR/data/processed/test"
mkdir -p "$MUCS_DIR/data/processed/blindtest"


# Process blindtest set
echo "Processing blindtest set..."
cp -f "$PYTHON_SCRIPT" "$MUCS_DIR/data/subtask2_blindtest_wReadme/hindi/"
cd "$MUCS_DIR/data/subtask2_blindtest_wReadme/hindi"
python "$(basename "$PYTHON_SCRIPT")" \
    --kaldi_dir "./files" \
    --output_audio_dir "$MUCS_DIR/data/processed/blindtest" \
    --sampling_rate 16000
cd "$SCRIPT_DIR"

# Process train set
echo "Processing train set..."
cp -f "$PYTHON_SCRIPT" "$MUCS_DIR/data/train/"
cd "$MUCS_DIR/data/train"
python "$(basename "$PYTHON_SCRIPT")" \
    --kaldi_dir "./transcripts" \
    --output_audio_dir "$MUCS_DIR/data/processed/train" \
    --sampling_rate 16000
cd "$SCRIPT_DIR"

# Process test set
echo "Processing test set..."
cp -f "$PYTHON_SCRIPT" "$MUCS_DIR/data/test/"
cd "$MUCS_DIR/data/test"
python "$(basename "$PYTHON_SCRIPT")" \
    --kaldi_dir "./transcripts" \
    --output_audio_dir "$MUCS_DIR/data/processed/test" \
    --sampling_rate 16000
cd "$SCRIPT_DIR"

echo "All datasets processed successfully!"
