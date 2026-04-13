#!/bin/bash
# PaddleOCR 常駐サーバーを起動するスクリプト

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

trap 'echo "OCR サーバーを停止します..."; exit 0' INT TERM

while true; do
    python -m automation.ocr_server "$@"
    EXIT_CODE=$?
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] OCR サーバーが終了しました (exit code: $EXIT_CODE)。3秒後に再起動します..."
    sleep 3
done
