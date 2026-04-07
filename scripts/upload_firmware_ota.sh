#!/bin/bash
# Upload firmware to ESP32 via OTA (WiFi) using espota.py.
#
# Usage:
#   ./scripts/upload_firmware_ota.sh                   # auto-discover via mDNS
#   ./scripts/upload_firmware_ota.sh 10.166.123.191    # specify IP directly
#
# Requirements:
#   - arduino-cli installed (brew install arduino-cli)
#   - ESP32 on the same WiFi network (hostname: ble-hid-bridge)
#   - wireless-input-bridge/wifi_config.h with correct credentials

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SKETCH_DIR="$PROJECT_ROOT/wireless-input-bridge"
BUILD_DIR="/tmp/ble-hid-bridge-build"
FQBN="esp32:esp32:m5stack_atoms3:USBMode=default,CDCOnBoot=cdc"
HOSTNAME="ble-hid-bridge"
ESP32_PACKAGE="$HOME/Library/Arduino15/packages/esp32/hardware/esp32"
ESPOTA="$(ls -d "$ESP32_PACKAGE"/*/tools/espota.py 2>/dev/null | sort -V | tail -1)"

if [ -z "$ESPOTA" ]; then
    echo "Error: espota.py not found. Make sure esp32 board package is installed."
    exit 1
fi

# ── Resolve target IP ───────────────────────────────────────────────────────
if [ "${1:-}" != "" ]; then
    TARGET_IP="$1"
    echo "Using specified IP: $TARGET_IP"
else
    echo "Resolving hostname '$HOSTNAME.local'..."
    TARGET_IP=$(ping -c1 -t3 "$HOSTNAME.local" 2>/dev/null | \
                grep -oE '[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+' | head -1) || true

    if [ -z "$TARGET_IP" ]; then
        echo "Error: Could not resolve '$HOSTNAME.local'."
        echo "Make sure the ESP32 is powered on and connected to WiFi."
        echo "Or pass the IP directly: $0 <ip-address>"
        exit 1
    fi
    echo "Resolved to: $TARGET_IP"
fi

# ── Compile ──────────────────────────────────────────────────────────────────
echo ""
echo "Compiling firmware..."
mkdir -p "$BUILD_DIR"
arduino-cli compile \
    --fqbn "$FQBN" \
    --output-dir "$BUILD_DIR" \
    "$SKETCH_DIR"

BIN="$BUILD_DIR/wireless-input-bridge.ino.bin"
if [ ! -f "$BIN" ]; then
    echo "Error: Compiled binary not found at $BIN"
    exit 1
fi

# ── Upload via OTA ───────────────────────────────────────────────────────────
echo ""
echo "Uploading to $TARGET_IP via OTA..."
python3 "$ESPOTA" -i "$TARGET_IP" -p 3232 -f "$BIN"

echo ""
echo "OTA upload complete. Device will reboot automatically."
