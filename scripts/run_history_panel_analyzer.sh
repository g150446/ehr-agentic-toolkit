#!/bin/bash
# Helper script to analyze the past-history panel on a saved screenshot

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "Error: Virtual environment not found. Run ./scripts/setup_automation.sh first."
    exit 1
fi

export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

python -m automation.history_panel_analyzer "$@"
