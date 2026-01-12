#!/bin/bash
# Setup script for Windows Login Automation
# This script ensures all dependencies are installed and configured

set -e  # Exit on error

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

echo "=========================================="
echo "EHR AI Bridge Toolkit - Automation Setup"
echo "=========================================="

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
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

# Install DocLayout-YOLO
echo "Installing DocLayout-YOLO..."
cd DocLayout-YOLO
pip install -e .
cd ..

# Install EasyOCR
echo "Installing EasyOCR..."
pip install easyocr

# Create .env if it doesn't exist
if [ ! -f ".env" ]; then
    echo "Creating .env file from template..."
    cp .env.example .env
    echo ""
    echo "⚠️  IMPORTANT: Edit .env file and set your configuration:"
    echo "   - WINDOWS_LOGIN_PASSWORD"
    echo "   - ESP32_DEVICE_NAME (if different)"
    echo ""
fi

# Create models directory
mkdir -p DocLayout-YOLO/models

# Download model if it doesn't exist
if [ ! -f "DocLayout-YOLO/models/doclayout.pt" ]; then
    echo "Downloading DocLayout-YOLO model..."
    python3 << 'EOF'
import sys
sys.path.insert(0, './DocLayout-YOLO')
try:
    from doclayout_yolo import YOLOv10
    print("Loading model from HuggingFace...")
    model = YOLOv10.from_pretrained("juliozhao/DocLayout-YOLO-DocStructBench")
    model.save("DocLayout-YOLO/models/doclayout.pt")
    print("✓ Model saved to DocLayout-YOLO/models/doclayout.pt")
except Exception as e:
    print(f"Warning: Could not download model: {e}")
    print("You can download it manually or skip model-based features.")
EOF
fi

# Create output directories
mkdir -p automation_outputs/screenshots
mkdir -p automation_outputs/logs

echo ""
echo "=========================================="
echo "✓ Setup complete!"
echo "=========================================="
echo ""
echo "To use automation:"
echo "  1. Activate venv: source venv/bin/activate"
echo "  2. Set PYTHONPATH: export PYTHONPATH=$PROJECT_ROOT:\$PYTHONPATH"
echo "  3. Edit .env file with your settings"
echo "  4. Run tests:"
echo "     python -m automation.windows_login --test-ble"
echo "     python -m automation.windows_login --test-capture"
echo ""
echo "Or use the helper script: ./run_automation.sh"
echo ""
