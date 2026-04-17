#!/bin/bash
# Send BACKSPACE key N times.
# If ble_server is already connected, send via the server socket.
# Otherwise connect directly to the BLE device.
# Usage: ./send_backspaces.sh [count]
# Default: 10 backspaces

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

COUNT="${1:-10}"

python - "$COUNT" <<'EOF'
import asyncio
import sys
import time
from automation.ble_client import BLEClient
from automation.ble_controller import BLEController
from automation.config import AutomationConfig

def send_via_server(count: int) -> None:
    client = BLEClient()
    print(f"BLE サーバー経由で BACKSPACE x{count} を送信します...")
    for i in range(count):
        if not client.press_key("backspace"):
            print(f"Error: Failed to send BACKSPACE via ble_server at press {i + 1}.")
            sys.exit(1)
        time.sleep(0.05)
    print(f"Sent via ble_server: BACKSPACE x{count}")


async def send_direct(count: int) -> None:
    config = AutomationConfig()
    ble = BLEController(
        device_name=config.esp32_device_name,
        service_uuid=config.ble_service_uuid,
        rx_char_uuid=config.ble_rx_char_uuid,
        tx_char_uuid=config.ble_tx_char_uuid
    )

    print(f"Connecting directly to {config.esp32_device_name}...")
    if not await ble.connect(timeout=15.0):
        print("Error: Failed to connect to BLE device.")
        sys.exit(1)
    print(f"Connected. Sending BACKSPACE x{count}...")

    try:
        await asyncio.sleep(0.3)
        for i in range(count):
            if not await ble.press_key("backspace"):
                print(f"Error: Failed to send BACKSPACE at press {i + 1}.")
                sys.exit(1)
            await asyncio.sleep(0.05)
        print(f"Sent directly: BACKSPACE x{count}")
    finally:
        await ble.disconnect()
        print("Disconnected.")


async def main():
    count = int(sys.argv[1])
    client = BLEClient()
    if client.is_server_running():
        send_via_server(count)
        return
    await send_direct(count)

asyncio.run(main())
EOF
