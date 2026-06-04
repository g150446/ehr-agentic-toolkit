#!/bin/bash
# Remote HTTP サーバーを起動するスクリプト
# ehr_controller の --last-prescription フローを HTTP API として公開する。
# 別マシンの remote_client/client.py から呼び出す際に使用。
#
# 環境変数:
#   REMOTE_SERVER_PORT      ポート番号 (デフォルト: 8765)
#   REMOTE_SERVER_API_KEY   Bearer トークン (未設定なら認証なし)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

if [ -d "venv" ]; then
    PYTHON="$PROJECT_ROOT/venv/bin/python"
else
    echo "Error: Virtual environment not found. Run ./scripts/setup_automation.sh first."
    exit 1
fi

export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

trap 'echo "Remote サーバーを停止します..."; exit 0' INT TERM

while true; do
    "$PYTHON" -m automation.remote_server "$@"
    EXIT_CODE=$?
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Remote サーバーが終了しました (exit code: $EXIT_CODE)。3秒後に再起動します..."
    sleep 3
done
