#!/bin/bash
# BLE 常駐サーバーを起動するスクリプト
# ESP32 BLE デバイスに接続し、/tmp/ble_server.sock でコマンドを受け付ける。
# ehr_input.py などのクライアントを実行する前に、別ターミナルで起動しておく。

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

python -m automation.ble_server "$@"
