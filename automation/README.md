# Automation Tools

This directory contains automation tools for HDMI capture, screen analysis, and Windows PC control.

## Tools

1. **HDMI Capture Stream Monitor** - Standalone real-time video monitoring with optional YOLO detection
2. **Windows Login Automation** - Automated login using HDMI capture and ESP32 BLE control
3. **Interactive Browser Assistant** - Terminal chat interface for remote Chrome browser control
4. **BLE Test CLI** - Interactive testing tool for ESP32 BLE keyboard and mouse

---

## BLE Test CLI

Interactive command-line tool for manually testing BLE keyboard and mouse commands with the ESP32 wireless input bridge. Provides a REPL-style interface for sending individual commands to test the BLE connection and input control.

### Quick Start

```bash
# Start interactive CLI
./scripts/run_ble_test.sh

# With custom device name
./scripts/run_ble_test.sh --device-name "My ESP32"

# With custom timeout
./scripts/run_ble_test.sh --timeout 15
```

### Interactive Commands

Once in the CLI, use these commands:

#### Connection Management
```
connect         - Scan and connect to ESP32
disconnect      - Disconnect from ESP32
status          - Show connection status and current mode
scan            - Scan for available BLE devices
```

#### Mode Switching
```
keyboard        - Switch to keyboard mode
mouse           - Switch to mouse mode
```

#### Keyboard Commands (requires keyboard mode)
```
type "text"     - Type text string
press <key>     - Press special key (enter, tab, esc, backspace, delete)
```

#### Mouse Commands (requires mouse mode)
```
move <x> <y>    - Move mouse relatively (positive=right/down, negative=left/up)
moveto <x> <y>  - Move to absolute position from top-left
click           - Left mouse click
scroll <amount> - Scroll wheel (positive=down, negative=up)
reset           - Reset cursor to origin (0, 0)
```

#### Utilities
```
raw <command>   - Send raw BLE UART command
help [command]  - Show help for all commands or specific command
quit / exit     - Exit the CLI
```

### Example Session

```
$ ./scripts/run_ble_test.sh

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BLE Keyboard & Mouse Test CLI
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Type 'help' for commands, 'quit' to exit

(BLE Test) connect
[INFO] Scanning for: BLE Mouse & Keyboard
[SUCCESS] Connected to ESP32 at XX:XX:XX:XX:XX:XX

(BLE Test) keyboard
[SUCCESS] Switched to keyboard mode

(BLE Test) type "Hello World"
[SUCCESS] Typed: Hello World

(BLE Test) press enter
[SUCCESS] Pressed: Enter

(BLE Test) mouse
[SUCCESS] Switched to mouse mode

(BLE Test) move 100 50
[SUCCESS] Moved: right 100, down 50

(BLE Test) click
[SUCCESS] Clicked

(BLE Test) status
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Connected:      Yes
Device:         BLE Mouse & Keyboard
Address:        XX:XX:XX:XX:XX:XX
Current Mode:   Mouse
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

(BLE Test) quit
Goodbye!
```

### Features

- **Interactive REPL interface** - Natural command-line interaction with history
- **Colored output** - Visual feedback with success/error/info messages
- **Connection status tracking** - Always know if you're connected and in what mode
- **Comprehensive help** - Built-in help for all commands
- **Error handling** - Clear error messages with helpful hints
- **Tab completion** - Command history with up/down arrows
- **Raw command mode** - Send any BLE UART command for debugging

### Use Cases

- **Test ESP32 BLE connection** - Verify ESP32 is advertising and accepting connections
- **Manual keyboard control** - Type text and press keys remotely
- **Manual mouse control** - Move cursor, click, and scroll remotely
- **Debug automation issues** - Test individual commands to isolate problems
- **Learn BLE protocol** - Understand the command format using raw mode
- **Integration testing** - Verify keyboard/mouse work before running automation

### Configuration

Uses the same `.env` configuration as other automation tools:

```bash
ESP32_DEVICE_NAME=BLE Mouse & Keyboard
BLE_SERVICE_UUID=6E400001-B5A3-F393-E0A9-E50E24DCCA9E
BLE_RX_CHAR_UUID=6E400002-B5A3-F393-E0A9-E50E24DCCA9E
BLE_TX_CHAR_UUID=6E400003-B5A3-F393-E0A9-E50E24DCCA9E
```

### Troubleshooting

**"Device not found"**
- Make sure ESP32 is powered on and advertising
- Check Bluetooth is enabled on your Mac
- Verify device name matches `ESP32_DEVICE_NAME` in `.env`
- Try the `scan` command to see available devices

**"Not connected to BLE device"**
- Run `connect` command first
- Check that ESP32 is not connected to another device
- Try disconnecting and reconnecting

**"Failed to send command"**
- Verify you're in the correct mode (keyboard/mouse)
- Check ESP32 is still connected (`status` command)
- Try reconnecting

---

## HDMI Capture Stream Monitor

Real-time video streaming tool for HDMI capture devices with optional DocLayout-YOLO detection overlay. Runs independently from the chat interface.

### Quick Start

```bash
# Basic streaming (5 FPS, raw mode)
./scripts/run_monitor_standalone.sh

# With detection enabled from start
./scripts/run_monitor_standalone.sh --detection-on

# Custom frame rate
./scripts/run_monitor_standalone.sh --fps 10

# Custom confidence threshold
./scripts/run_monitor_standalone.sh --confidence 0.3 --detection-on
```

### Keyboard Controls

| Key | Action |
|-----|--------|
| **Q** / **ESC** | Quit application |
| **D** | Toggle YOLO detection ON/OFF |
| **S** | Save screenshot |
| **F** | Toggle FPS counter |
| **H** | Toggle help overlay |
| **+** | Increase confidence threshold |
| **-** | Decrease confidence threshold |

### Features

- **Real-time streaming** at configurable frame rate (default: 5 FPS)
- **Toggle detection mode** - Switch between raw video and YOLO detection overlay on the fly
- **Performance optimized** - Persistent VideoCapture, lazy model loading, low CPU usage
- **Status overlay** - Shows mode, FPS, device info, detection count, and confidence
- **Screenshot capture** - Save any frame with timestamp
- **Dynamic confidence** - Adjust detection threshold in real-time with +/- keys

### Use Cases

- **Debug HDMI capture** - Verify MiraBox device is working and receiving video
- **Monitor Windows login screen** - Live view of the screen being automated
- **Test YOLO detection** - Visualize what the automation system sees
- **Capture screenshots** - Save frames for analysis or debugging
- **Performance testing** - Measure actual FPS and detection performance

### Command-Line Options

```bash
python -m automation.monitor_standalone \
  --device 0              # Video capture device index (default: 0)
  --fps 5                 # Target frame rate (default: 5.0)
  --detection-on          # Enable YOLO detection from start
  --confidence 0.2        # Detection confidence threshold
  --imgsz 1024           # YOLO image size
  --output-dir ./outputs  # Screenshot directory
  --debug                 # Enable debug logging
```

### Output

Screenshots are saved to `monitor_outputs/` with timestamps:
```
monitor_outputs/
├── monitor_screenshot_20260112_160010.jpg
├── monitor_screenshot_20260112_160025.jpg
└── ...
```

Logs are saved to `automation_outputs/logs/monitor_TIMESTAMP.log`

---

## Interactive Browser Assistant

Terminal-based chat assistant that controls a remote Windows PC's Chrome browser via HDMI capture and ESP32 BLE keyboard/mouse emulation.

### Quick Start

```bash
# Start interactive assistant
./scripts/run_pc_controller.sh

# With custom device
./scripts/run_pc_controller.sh --device 1

# Debug mode
./scripts/run_pc_controller.sh --debug
```

For HDMI capture monitor, use the standalone script:
```bash
./scripts/run_monitor_standalone.sh
```

### Features

- **Terminal Chat Interface** - Natural language commands in your terminal
- **Dual YOLO Models** - Switch between DocLayout-YOLO and YOLOv11 UI detection
- **Browser Automation** - Open Chrome, navigate to URLs, click elements
- **Screen Analysis** - Analyze current screen with either model
- **Screenshot Capture** - Save frames for analysis
- **HDMI Capture Monitor** - Real-time video streaming with detection overlay

### Chat Commands

| Command | Description | Example |
|---------|-------------|---------|
| **open chrome** | Launch Chrome browser | `open chrome` |
| **goto <url>** | Navigate to URL | `goto google.com` |
| **click address bar** | Click the address bar | `click address bar` |
| **switch to doclayout** | Use DocLayout-YOLO model | `switch to doclayout` |
| **switch to ui detection** | Use YOLOv11 UI model | `switch to ui detection` |
| **analyze** | Analyze current screen | `analyze` |
| **capture** | Save screenshot | `capture` |
| **help** | Show all commands | `help` |
| **quit** | Exit assistant | `quit` |

### Chat Session Example

```
🤖 Interactive Browser Assistant
Control remote Chrome browser via chat commands.
Type 'help' for available commands.

[doclayout] > open chrome
✅ Chrome opened

[doclayout] > switch to ui detection
✅ Switched to ui-detection model

[ui-detection] > goto google.com
✅ Navigated to https://google.com

[ui-detection] > analyze
📊 Detected 8 elements:
  1. text (confidence: 0.85)
  2. button (confidence: 0.78)
  3. input (confidence: 0.92)
  ...

[ui-detection] > quit
✅ Goodbye!
```

### How It Works

1. **Initialization** - Load models (DocLayout-YOLO default), connect to ESP32 BLE
2. **Chat Loop** - Accept commands via terminal prompt
3. **Command Parsing** - Parse natural language into structured commands
4. **Model Switching** - Switch between DocLayout-YOLO and YOLOv11 on demand
5. **Browser Control** - Detect UI elements (address bar) and send BLE commands
6. **Navigation** - Click address bar, type URL, press Enter

### YOLO Models

**DocLayout-YOLO** (default):
- Document layout detection
- Tables, text blocks, figures, titles
- Best for document analysis
- Also used by standalone monitor for screen detection

**YOLOv8 UI Detection** (`foduucom/web-form-ui-field-detection`):
- Web form and UI field detection
- Text inputs, buttons, radio buttons, checkboxes, email/password fields
- Best for browser automation
- Uses ultralyticsplus library

### Output

Screenshots are saved to `automation_outputs/chat_screenshots/` with timestamps:
```
automation_outputs/chat_screenshots/
├── chat_capture_20260112_170530.jpg
├── chat_capture_20260112_170545.jpg
└── ...
```

Logs are saved to `automation_outputs/logs/browser_assistant_TIMESTAMP.log`

### Prerequisites

- Windows PC connected via HDMI to capture device
- ESP32 running BLE keyboard/mouse firmware
- MiraBox or compatible HDMI capture device
- Chrome browser installed on remote PC

### Troubleshooting

**"Failed to load YOLOv8 UI detection model"**

Install ultralyticsplus (required for web form UI detection):
```bash
source venv/bin/activate
pip install ultralyticsplus>=0.0.28 ultralytics>=8.0.0
```

Or run the setup script again:
```bash
./scripts/setup_automation.sh
```

**"Address bar not detected"**

Try switching to UI detection model first:
```
[doclayout] > switch to ui detection
[ui-detection] > goto google.com
```

---

## Windows Login Automation

Automates Windows PC login using HDMI screen capture and ESP32 BLE keyboard/mouse emulation.

## Quick Start

### 1. Setup (First Time)

```bash
# From project root
./scripts/setup_automation.sh

# Edit .env file with your password
nano .env  # Set WINDOWS_LOGIN_PASSWORD
```

### 2. Run Tests

```bash
# Test screen capture
./scripts/run_automation.sh --test-capture

# Test ESP32 BLE connection
./scripts/run_automation.sh --test-ble
```

### 3. Run Full Automation

```bash
# Debug mode (step-by-step with pauses)
./scripts/run_automation.sh --password YOUR_PASSWORD --debug

# Normal mode (automatic)
./scripts/run_automation.sh --password YOUR_PASSWORD

# With monitor window (see live video + automation progress) ⭐ NEW
./scripts/run_automation.sh --password YOUR_PASSWORD --monitor-mode

# Monitor mode + debug (best for testing/troubleshooting)
./scripts/run_automation.sh --password YOUR_PASSWORD --monitor-mode --debug
```

### Monitor Mode

The `--monitor-mode` flag displays a **live window** showing what the automation sees in real-time:

**Features:**
- 📺 **Real-time video** from HDMI capture device
- 📊 **Status overlay** showing current phase and progress (Phase 1/5, 2/5, etc.)
- 🎮 **Interactive controls**:
  - Press **'q'** to quit automation early
  - Press **'s'** to save screenshot manually
- 🔄 **Live updates** as automation progresses through each phase

**When to use:**
- ✅ **Testing and debugging** - See exactly what the automation detects
- ✅ **First-time setup** - Verify everything is working correctly
- ✅ **Troubleshooting** - Watch the automation in action to identify issues
- ❌ **Production use** - Not needed once automation is working reliably

**Important:** Monitor mode and `run_monitor.sh` both access the same HDMI capture device - use only one at a time.

## Manual Setup (Alternative)

If you prefer manual control:

```bash
# Activate venv
source venv/bin/activate

# Set PYTHONPATH
export PYTHONPATH=/Users/g150446/gitdir/ehr-ai-bridge-toolkit:$PYTHONPATH

# Run automation
python -m automation.windows_login --test-capture
```

## Configuration

Edit `.env` file in project root:

```bash
# ESP32 BLE Configuration
ESP32_DEVICE_NAME=BLE Mouse & Keyboard

# Video Capture
CAPTURE_DEVICE_INDEX=0
CAPTURE_WIDTH=1920
CAPTURE_HEIGHT=1080

# Windows Login
WINDOWS_LOGIN_PASSWORD=your_password
LOGIN_DEBUG_MODE=true
LOGIN_AUTO_VERIFY=true

# DocLayout-YOLO
DOCLAYOUT_MODEL_PATH=./DocLayout-YOLO/models/doclayout.pt
DETECTION_CONFIDENCE=0.2
DETECTION_IMAGE_SIZE=1024
DETECTION_DEVICE=auto
```

## How It Works

### 5-Phase Pipeline

1. **Initialization**: Load DocLayout-YOLO model, EasyOCR, connect to ESP32
2. **Screen Capture**: Capture Windows login screen via HDMI
3. **Screen Analysis**: Detect UI elements using YOLO + OCR
4. **Input Control**: Send password via ESP32 BLE keyboard emulation
5. **Verification**: Verify login success by screen change detection

### Architecture

```
HDMI Capture (MiraBox)
        ↓
  Mac (Claude)
        ↓
DocLayout-YOLO + EasyOCR
        ↓
    Analysis
        ↓
  BLE Commands
        ↓
   ESP32 Module
        ↓
  USB HID (Keyboard/Mouse)
        ↓
   Windows PC
```

## Troubleshooting

### "ModuleNotFoundError: No module named 'automation'"

Set PYTHONPATH:
```bash
export PYTHONPATH=/Users/g150446/gitdir/ehr-ai-bridge-toolkit:$PYTHONPATH
```

Or use the helper script: `./scripts/run_automation.sh`

### "No module named 'doclayout_yolo'"

Install DocLayout-YOLO:
```bash
source venv/bin/activate
cd DocLayout-YOLO
pip install -e .
cd ..
```

### "Configuration error: DocLayout-YOLO model not found"

Download model:
```bash
python3 -c "
from doclayout_yolo import YOLOv10
model = YOLOv10.from_pretrained('juliozhao/DocLayout-YOLO-DocStructBench')
model.save('DocLayout-YOLO/models/doclayout.pt')
"
```

### ESP32 Not Connecting

1. Check Bluetooth is enabled on Mac
2. Verify ESP32 is powered and advertising
3. Check device name in .env matches ESP32's advertised name
4. Try: `./scripts/run_automation.sh --test-ble`

### Screen Capture Not Working

1. Check MiraBox is connected and powered
2. Verify Windows PC HDMI output is connected to MiraBox
3. Check device index (usually 0, try 1 or 2 if not working)
4. Try: `./scripts/run_automation.sh --test-capture`

## Output Files

All outputs are saved to `automation_outputs/`:

- `screenshots/`: Captured screens and debug visualizations
- `logs/`: Detailed logs with timestamps

## Advanced Usage

### Debug Mode

Step-by-step execution with pauses:
```bash
./scripts/run_automation.sh --debug
```

### Skip Verification

Don't verify login success:
```bash
./scripts/run_automation.sh --no-verify
```

### Custom .env Location

```bash
./scripts/run_automation.sh --env-file /path/to/.env
```

## Development

### Module Structure

- `config.py`: Configuration and .env loading
- `ble_controller.py`: ESP32 BLE communication
- `ble_test_cli.py`: Interactive BLE testing CLI tool
- `screen_analyzer.py`: DocLayout-YOLO + EasyOCR integration
- `model_manager.py`: Multi-model management (DocLayout-YOLO + YOLOv11)
- `utils.py`: Logging, debugging, progress tracking
- `monitor_stream.py`: Original HDMI capture stream monitor with YOLO detection (legacy)
- `monitor_standalone.py`: Standalone HDMI capture monitor (recommended)
- `windows_login.py`: Main automation pipeline
- `browser_assistant.py`: Interactive browser automation chat tool

### Adding Features

1. Edit relevant module in `automation/`
2. Test with `--debug` flag
3. Check logs in `automation_outputs/logs/`
