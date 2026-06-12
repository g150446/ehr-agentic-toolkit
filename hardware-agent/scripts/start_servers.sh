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

# ── ndlocr-lite インストール確認 ──────────────────────────────
echo "[$(date '+%Y-%m-%d %H:%M:%S')] ndlocr-lite のインストールを確認中..."
if ! "$PYTHON" -c "import deim" 2>/dev/null; then
    echo ""
    echo "Error: ndlocr-lite がインストールされていません。"
    echo ""
    echo "以下のコマンドで自動インストールしてください："
    echo "  ./scripts/install_ndlocr.sh"
    echo ""
    echo "(git clone + 依存パッケージ + モデルファイルのダウンロードを自動実行します)"
    echo ""
    exit 1
fi
echo "[$(date '+%Y-%m-%d %H:%M:%S')] ndlocr-lite のインストールを確認しました。"
# ──────────────────────────────────────────────────────────────

# ── HDMI キャプチャデバイス ウォームアップ ──────────────────────
# Mac Mini コールドスタート時、MiraBox が HDMI 信号をロックするまで待機する。
# QT Player の「新規ムービー収録」と同等の AVCaptureSession 初期化を行う。
echo "[$(date '+%Y-%m-%d %H:%M:%S')] HDMI キャプチャデバイスをウォームアップ中..."
"$PYTHON" "$SCRIPT_DIR/warmup_hdmi.py"
WARMUP_EXIT=$?
if [ $WARMUP_EXIT -ne 0 ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 警告: HDMI ウォームアップがタイムアウトしました。最初のキャプチャが失敗する可能性があります。"
else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] HDMI ウォームアップ完了。"
fi
# ──────────────────────────────────────────────────────────────

BLE_LOOP_PID=""
REMOTE_PID=""

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
    [ -n "$REMOTE_PID" ] && kill "$REMOTE_PID" 2>/dev/null
    kill -- -"$BLE_LOOP_PID" 2>/dev/null   # BLE プロセスグループ全体を終了
    wait 2>/dev/null
    echo "停止しました"
    exit 0
}
trap cleanup INT TERM

# 既存プロセスが残っていればポートを解放する
_PORT="${REMOTE_SERVER_PORT:-8765}"
_EXISTING=$(lsof -ti "tcp:$_PORT" 2>/dev/null)
if [ -n "$_EXISTING" ]; then
    echo "ポート $_PORT が使用中です (PID: $_EXISTING)。停止します..."
    kill $_EXISTING 2>/dev/null
    sleep 1
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] BLE サーバー起動 (PID: $BLE_LOOP_PID)"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Remote サーバー起動中... (port: $_PORT)"
echo "停止するには Ctrl+C を押してください。"

# Remote サーバーをバックグラウンドで実行し wait する。
# こうすることで bash がターミナルのフォアグラウンドとなり、
# Ctrl+C の SIGINT を bash が受け取れるようになる（set -m 環境での回避策）。
while true; do
    "$PYTHON" -m automation.remote_server &
    REMOTE_PID=$!
    wait "$REMOTE_PID"
    EXIT_CODE=$?
    REMOTE_PID=""
    # 正常終了（Ctrl+C で Python が KeyboardInterrupt を処理した場合）はループを抜ける
    if [ $EXIT_CODE -eq 0 ]; then
        cleanup
    fi
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Remote サーバーが終了しました (exit: $EXIT_CODE)。3秒後に再起動します..."
    sleep 3
done
