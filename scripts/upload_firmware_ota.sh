#!/bin/bash
# Upload firmware to ESP32 via OTA (WiFi) using arduino-cli.
#
# Usage:
#   ./scripts/upload_firmware_ota.sh                   # auto-discover device
#   ./scripts/upload_firmware_ota.sh 192.168.1.42      # specify IP directly
#
# Requirements:
#   - arduino-cli installed (brew install arduino-cli)
#   - ESP32 on the same WiFi network (hostname: ble-hid-bridge)
#   - wireless-input-bridge/wifi_config.h with correct credentials

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SKETCH_DIR="$PROJECT_ROOT/wireless-input-bridge"
FQBN="esp32:esp32:m5stack_atoms3:USBMode=default,CDCOnBoot=cdc"
HOSTNAME="ble-hid-bridge"

# ── Resolve target IP ───────────────────────────────────────────────────────
if [ "${1:-}" != "" ]; then
    TARGET_IP="$1"
    echo "Using specified IP: $TARGET_IP"
else
    echo "Resolving hostname '$HOSTNAME'..."
    TARGET_IP=$(dns-sd -G v4 "$HOSTNAME.local" 2>/dev/null | \
                awk '/Add/{print $6; exit}' &
                PID=$!
                sleep 3
                kill $PID 2>/dev/null || true) || true

    # Fallback: try ping-based resolution
    if [ -z "$TARGET_IP" ]; then
        TARGET_IP=$(ping -c1 -t2 "$HOSTNAME.local" 2>/dev/null | \
                    grep -oE '[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+' | head -1) || true
    fi

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
arduino-cli compile \
    --fqbn "$FQBN" \
    "$SKETCH_DIR"

# ── Upload via OTA ───────────────────────────────────────────────────────────
echo ""
echo "Uploading to $TARGET_IP via OTA..."
arduino-cli upload \
    --fqbn "$FQBN" \
    --port "$TARGET_IP" \
    --protocol network \
    "$SKETCH_DIR"

echo ""
echo "OTA upload complete. Device will reboot automatically."
