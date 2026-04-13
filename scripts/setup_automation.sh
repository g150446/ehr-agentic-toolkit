#!/bin/bash
# Setup script for Windows Login Automation
# This script ensures all dependencies are installed and configured

set -e  # Exit on error

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

echo "=========================================="
echo "EHR AI Bridge Toolkit - Automation Setup"
echo "=========================================="

if command -v python3.12 >/dev/null 2>&1; then
    PYTHON_BIN="python3.12"
elif command -v python3.11 >/dev/null 2>&1; then
    PYTHON_BIN="python3.11"
else
    PYTHON_BIN="python3"
fi

echo "Using Python interpreter: $PYTHON_BIN"

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    "$PYTHON_BIN" -m venv venv
fi

# Activate venv
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Create .env if it doesn't exist
if [ ! -f ".env" ]; then
    echo "Creating .env file from template..."
    cp .env.example .env
    echo ""
    echo "⚠️  IMPORTANT: Edit .env file and set your configuration:"
    echo "   - ESP32_DEVICE_NAME (if different)"
    echo ""
fi

# Create output directories
mkdir -p automation_outputs/screenshots
mkdir -p automation_outputs/logs
mkdir -p automation_outputs/history_panel_analysis

echo ""
echo "=========================================="
echo "✓ Setup complete!"
echo "=========================================="
echo ""
echo "To use automation:"
echo "  1. Activate venv: source venv/bin/activate"
echo "  2. Set PYTHONPATH: export PYTHONPATH=$PROJECT_ROOT:\$PYTHONPATH"
echo "  3. Edit .env file with your settings"
echo "  4. Start the MLX VLM server when using click_history / mlx_vlm_history:"
echo "     ./scripts/start_mlx_vlm_server.sh qwen"
echo "  5. Run tests:"
echo "     ./scripts/run_ble_test.sh"
echo "     ./scripts/run_monitor.sh"
echo ""
