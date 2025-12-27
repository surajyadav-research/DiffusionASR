#!/bin/bash

# Get the script directory and find the repo root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Load environment variables from .env file in repo root
if [ -f "$REPO_ROOT/.env" ]; then
    source "$REPO_ROOT/.env"
else
    echo "Error: .env file not found in repository root ($REPO_ROOT)!"
    echo "Please create a .env file in the repository root with your Hugging Face token:"
    echo "HF_TOKEN=your_token_here"
    exit 1
fi

# Check if token is set
if [ -z "$HF_TOKEN" ]; then
    echo "Error: HF_TOKEN not set in .env file!"
    exit 1
fi

# Download model files with authentication
wget --header="Authorization: Bearer $HF_TOKEN" https://huggingface.co/meta-llama/Llama-3.2-3B/resolve/main/model-00001-of-00002.safetensors
wget --header="Authorization: Bearer $HF_TOKEN" https://huggingface.co/meta-llama/Llama-3.2-3B/resolve/main/model-00002-of-00002.safetensors