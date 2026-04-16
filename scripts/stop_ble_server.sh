#!/bin/bash
# BLE 常駐サーバーを停止するスクリプト
# start_ble_server.sh で起動したサーバーを停止します。

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
START_SCRIPT="$SCRIPT_DIR/start_ble_server.sh"

# start_ble_server.sh プロセスを探して終了
echo "BLE サーバーを停止しています..."
PIDS=$(pgrep -f "start_ble_server\.sh" | grep -v $$)

if [ -z "$PIDS" ]; then
    echo "BLE サーバーは起動していません。"
    exit 0
fi

for PID in $PIDS; do
    # 自分自身のスクリプトは除外
    if [ "$PID" -eq $$ ]; then
        continue
    fi
    echo "PID $PID の BLE サーバープロセスを停止しています..."
    kill "$PID" 2>/dev/null
done

# プロセスが終了するまで待つ（最大10秒）
for i in {1..10}; do
    RUNNING=0
    for PID in $PIDS; do
        if ps -p "$PID" > /dev/null 2>&1; then
            RUNNING=1
            break
        fi
    done
    if [ $RUNNING -eq 0 ]; then
        echo "BLE サーバーが停止しました。"
        exit 0
    fi
    sleep 1
done

# まだ動いている場合は強制終了
echo "警告: 一部のプロセスが応答しません。強制終了します..."
for PID in $PIDS; do
    if ps -p "$PID" > /dev/null 2>&1; then
        kill -9 "$PID" 2>/dev/null
        echo "PID $PID を強制終了しました。"
    fi
done

echo "BLE サーバーの停止を完了しました。"