#!/bin/bash
# Flash the wireless-input-bridge firmware to the ESP32.
# The ESP32 must be connected via USB before running this script.
# Usage: ./scripts/flash_firmware.sh [PORT]
#   PORT: serial port (default: auto-detect)

set -euo pipefail

SKETCH="wireless-input-bridge/wireless-input-bridge.ino"
FQBN="esp32:esp32:esp32s3:USBMode=default"

PORT="${1:-}"
if [ -z "$PORT" ]; then
    PORT=$(ls /dev/cu.usb* /dev/tty.usb* 2>/dev/null | head -1 || true)
    if [ -z "$PORT" ]; then
        echo "ERROR: ESP32 not found. Connect via USB and try again." >&2
        echo "  Available ports:" >&2
        ls /dev/cu.* 2>/dev/null | grep -v Bluetooth | grep -v wlan | grep -v debug | head -10 >&2
        exit 1
    fi
fi

echo "==> Compiling firmware for $FQBN ..."
arduino-cli compile --fqbn "$FQBN" "$SKETCH"

echo "==> Uploading to $PORT ..."
arduino-cli upload --fqbn "$FQBN" --port "$PORT" "$SKETCH"

echo "==> Done. Firmware flashed to $PORT"
