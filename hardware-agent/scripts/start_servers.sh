#!/bin/bash
# BLE サーバーと Remote HTTP サーバーを一括起動するスクリプト
#
# venv を有効化し、BLE サーバーをバックグラウンドで、
# Remote サーバーをフォアグラウンドで起動する。
# Ctrl+C で両サーバーを同時に停止できる。
#
# 環境変数:
#   REMOTE_SERVER_PORT      ポート番号 (デフォルト: 8765)
#   REMOTE_SERVER_API_KEY   Bearer トークン (未設定なら認証なし)

set -m  # ジョブ制御を有効化 → バックグラウンドジョブに独立したプロセスグループを付与

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

if [ -d "venv" ]; then
    # shellcheck source=/dev/null
    source "$PROJECT_ROOT/venv/bin/activate"
    PYTHON="$PROJECT_ROOT/venv/bin/python"
else
    echo "Error: Virtual environment not found. Run ./scripts/setup_automation.sh first."
    exit 1
fi

export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# BLE 再起動ループをバックグラウンドで実行
_ble_loop() {
    while true; do
        "$PYTHON" -m automation.ble_server
        EXIT_CODE=$?
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] BLE サーバーが終了しました (exit: $EXIT_CODE)。3秒後に再起動します..."
        sleep 3
    done
}
_ble_loop &
BLE_LOOP_PID=$!

# Ctrl+C / SIGTERM で両サーバーを停止
cleanup() {
    echo ""
    echo "サーバーを停止します..."
    kill -- -"$BLE_LOOP_PID" 2>/dev/null   # BLE プロセスグループ全体を終了
    wait "$BLE_LOOP_PID" 2>/dev/null
    exit 0
}
trap cleanup INT TERM

echo "[$(date '+%Y-%m-%d %H:%M:%S')] BLE サーバー起動 (PID: $BLE_LOOP_PID)"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Remote サーバー起動中... (port: ${REMOTE_SERVER_PORT:-8765})"
echo "停止するには Ctrl+C を押してください。"

# Remote サーバーをフォアグラウンドで実行（再起動ループ付き）
while true; do
    "$PYTHON" -m automation.remote_server
    EXIT_CODE=$?
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Remote サーバーが終了しました (exit: $EXIT_CODE)。3秒後に再起動します..."
    sleep 3
done
