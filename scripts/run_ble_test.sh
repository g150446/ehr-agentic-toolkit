#!/bin/bash
# Helper script to run BLE Test CLI
# Automatically sets up environment and runs the interactive CLI

# Get project root (parent directory of scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Activate venv if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "Error: Virtual environment not found. Run ./scripts/setup_automation.sh first."
    exit 1
fi

# Set PYTHONPATH
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Run BLE test CLI with all passed arguments
python -m automation.ble_test_cli "$@"
