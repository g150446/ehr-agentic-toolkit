#!/bin/bash
# Automatically connects to BLE device and sends: SPACE -> "zenzai" -> ENTER

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

python - <<'EOF'
import asyncio
import sys
from automation.ble_controller import BLEController
from automation.config import AutomationConfig

async def main():
    config = AutomationConfig()
    ble = BLEController(
        device_name=config.esp32_device_name,
        service_uuid=config.ble_service_uuid,
        rx_char_uuid=config.ble_rx_char_uuid,
        tx_char_uuid=config.ble_tx_char_uuid
    )

    print(f"Connecting to {config.esp32_device_name}...")
    if not await ble.connect(timeout=15.0):
        print("Error: Failed to connect to BLE device.")
        sys.exit(1)
    print("Connected.")

    try:
        await asyncio.sleep(0.3)

        if not await ble.press_key("space"):
            print("Error: Failed to send SPACE key.")
            sys.exit(1)
        print("Sent: SPACE")
        await asyncio.sleep(0.2)

        if not await ble.type_text("zenzai"):
            print("Error: Failed to type text.")
            sys.exit(1)
        print("Sent: zenzai")
        await asyncio.sleep(0.2)

        if not await ble.press_key("enter"):
            print("Error: Failed to send ENTER key.")
            sys.exit(1)
        print("Sent: ENTER")

    finally:
        await ble.disconnect()
        print("Disconnected.")

asyncio.run(main())
EOF
