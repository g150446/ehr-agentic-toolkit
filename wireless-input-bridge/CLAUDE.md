# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an Arduino sketch for ESP32-S2/S3 that creates a USB HID mouse controlled via Bluetooth Low Energy (BLE) UART. The device advertises as "BLE Mouse Control" and accepts text commands over BLE to perform mouse operations (clicks, movement, scrolling) on the connected computer.

## Hardware Requirements

- ESP32-S2 or ESP32-S3 with Native USB support
- Arduino IDE with ESP32 board support installed
- Must be configured with USB in standard mode (not OTG mode)

## Build and Upload

This project uses Arduino IDE. To build and upload:

1. Open `BLE_UART_MouseControl.ino` in Arduino IDE
2. Select the appropriate ESP32-S2 or ESP32-S3 board from Tools > Board
3. Ensure USB Mode is set to "USB-OTG (TinyUSB)" or similar (NOT "Hardware CDC and JTAG")
4. Select the correct port from Tools > Port
5. Click Upload button or use Sketch > Upload

Serial Monitor (115200 baud) shows connection status and received commands.

## Architecture

### Single-File Design

The entire application is contained in `BLE_UART_MouseControl.ino`. There are no separate header or source files.

### Key Components

1. **BLE UART Service**: Uses standard Nordic UART Service UUIDs
   - Service: `6E400001-B5A3-F393-E0A9-E50E24DCCA9E`
   - RX Characteristic (WRITE): `6E400002-B5A3-F393-E0A9-E50E24DCCA9E` - receives commands
   - TX Characteristic (NOTIFY): `6E400003-B5A3-F393-E0A9-E50E24DCCA9E` - unused but required for standard UART service

2. **USB HID Mouse Emulation**: Uses ESP32's native USB to emulate a standard USB mouse
   - `USBHIDMouse` library provides `Mouse.click()`, `Mouse.move(x, y, scroll)` functions
   - The ESP32 appears to the computer as a physical USB mouse

3. **Command Processing**: `processCommand()` function at BLE_UART_MouseControl.ino:62
   - Parses text commands received via BLE
   - Supported formats: `click`, `up:N`, `down:N`, `left:N`, `right:N`, `scroll:N`
   - Commands are case-sensitive and use colon-delimited parameters

### Control Flow

1. `setup()`: Initializes USB mouse, creates BLE server, service, and characteristics, starts advertising
2. `MyCallbacks::onWrite()`: Triggered when BLE client writes to RX characteristic, calls `processCommand()`
3. `processCommand()`: Parses command string and executes corresponding mouse action
4. `loop()`: Handles BLE connection state changes and restarts advertising when disconnected

## Command Protocol

Commands are sent as ASCII strings via BLE UART:
- `click` - Left mouse button click
- `up:10` - Move cursor up 10 pixels (negative Y)
- `down:10` - Move cursor down 10 pixels (positive Y)
- `left:10` - Move cursor left 10 pixels (negative X)
- `right:10` - Move cursor right 10 pixels (positive X)
- `scroll:5` - Scroll down 5 units (positive = down, negative = up)

The third parameter of `Mouse.move(x, y, scroll)` controls scroll wheel.

## Testing

Connect via BLE UART apps:
- Android: Serial Bluetooth Terminal, nRF Connect
- iOS: LightBlue, nRF Connect

Commands can be tested through Serial Monitor during development, but actual mouse control requires BLE connection.

## Platform Constraints

The code uses preprocessor directives (BLE_UART_MouseControl.ino:22-28) to ensure compilation only occurs on ESP32 boards with native USB in the correct mode. If `ARDUINO_USB_MODE` is not defined or equals 1 (OTG mode), the sketch compiles to empty `setup()`/`loop()` functions with warnings/errors.
