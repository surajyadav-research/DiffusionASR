#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Path to the checkpoints directory
CHECKPOINTS_DIR="$PROJECT_ROOT/checkpoints"

# Check if checkpoints directory exists
if [ ! -d "$CHECKPOINTS_DIR" ]; then
    echo "Error: Checkpoints directory not found at $CHECKPOINTS_DIR"
    exit 1
fi

echo "Starting cleanup of checkpoints directory: $CHECKPOINTS_DIR"
echo "----------------------------------------"

# Run the cleanup script
python "$PROJECT_ROOT/cleanup.py" "$CHECKPOINTS_DIR"

echo "----------------------------------------"
echo "Cleanup completed!"
