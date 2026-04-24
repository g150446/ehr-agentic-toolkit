# CLAUDE.md

This file provides guidance to AI coding assistants (Claude Code, GitHub Copilot CLI, etc.) when working with this Arduino sketch.

## Project Overview

ESP32-S3 (M5AtomS3U) sketch that bridges BLE UART commands to USB HID mouse and keyboard actions.
A Mac sends BLE UART commands → ESP32 → USB HID → Windows PC.

## Build and Upload

### FQBN (Critical Settings)

```
esp32:esp32:m5stack_atoms3:USBMode=default,CDCOnBoot=default
```

**`CDCOnBoot=default` (Disabled) is mandatory.** Do NOT change to `CDCOnBoot=cdc`.

> **Why**: `CDCOnBoot=cdc` (Enabled) adds a USB CDC serial interface, making the device a USB
> Composite Device. Windows then fails to enumerate the HID mouse/keyboard interfaces, especially
> after OTA reboots. This was confirmed to break HID recognition; disabling CDC fixes it.
> OTA (WiFi) still works without CDC. The only loss is USB serial output.

### OTA Update (normal workflow)

```bash
./scripts/upload_firmware_ota.sh           # auto-discover via mDNS
./scripts/upload_firmware_ota.sh <ip>      # specify IP directly
```

### USB Flash (recovery only)

1. Put device into bootloader mode: hold BOOT button, press+release RESET, release BOOT
2. Port changes to `/dev/cu.usbmodem*`
3. Run:
   ```bash
   arduino-cli compile --fqbn "esp32:esp32:m5stack_atoms3:USBMode=default,CDCOnBoot=default" \
     --output-dir /tmp/ble-hid-bridge-build wireless-input-bridge
   arduino-cli upload --fqbn "esp32:esp32:m5stack_atoms3:USBMode=default,CDCOnBoot=default" \
     --port /dev/cu.usbmodem<XXXX> --input-dir /tmp/ble-hid-bridge-build wireless-input-bridge
   ```

## Architecture

```
Mac
 └─(BLE UART)─► ESP32 M5AtomS3U ─(USB HID)─► Windows PC
                      │
                      └─(WiFi OTA)─► Firmware update
```

- `USBHIDAbsoluteMouse` only (no relative mouse). Absolute and relative mouse share a
  `static bool initialized` in the base class; only one can be registered.
- Firmware tracks cursor position (`g_abs_x`, `g_abs_y`) in HID units (0–32767).
- `moveto:X,Y` sets absolute HID position; `move:DX,DY` adds delta to tracked position.

## Key Files

- `wireless-input-bridge.ino` — main sketch
- `wifi_config.h` — WiFi credentials (gitignored); copy from `wifi_config.h.example`
- `../scripts/upload_firmware_ota.sh` — OTA upload script (use this for all normal updates)

## Known Issues / Constraints

- **CDCOnBoot must be Disabled**: See FQBN section above.
- Only one mouse instance can be registered (static initializer limitation). Current firmware
  uses `USBHIDAbsoluteMouse` exclusively.
- `wifi_config.h` is gitignored. After cloning, copy the example and fill in credentials.
- USB serial (`Serial.print`) has no output when CDCOnBoot is disabled. Use BLE TX characteristic
  or WiFi serial for debugging if needed.
- **`type:` command uses per-character 30ms delay**: `Keyboard.print(text)` sends all characters
  without delay, which breaks Windows Japanese IME romaji-to-kana conversion (e.g., "tesuto"
  becomes "テsウtオ" instead of "テスト"). The firmware instead calls `Keyboard.write()` per
  character with `delay(30)` so the IME has time to process each romaji pair. Do NOT revert to
  `Keyboard.print()` or remove the delay.
