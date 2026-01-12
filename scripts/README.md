# Helper Scripts

This directory contains helper scripts for the EHR AI Bridge Toolkit automation tools.

## Scripts

### `setup_automation.sh`

**Purpose**: First-time setup for the automation environment

**What it does:**
- Creates Python virtual environment (`venv/`)
- Installs all dependencies from `requirements.txt`
- Installs DocLayout-YOLO in editable mode
- Installs EasyOCR
- Creates `.env` file from template if it doesn't exist
- Downloads DocLayout-YOLO model
- Creates output directories

**Usage:**
```bash
./scripts/setup_automation.sh
```

**When to use:** Run this once before using any automation tools.

---

### `run_automation.sh`

**Purpose**: Run Windows login automation

**What it does:**
- Activates virtual environment
- Sets PYTHONPATH
- Runs `automation.windows_login` module

**Usage:**
```bash
# Test BLE connection
./scripts/run_automation.sh --test-ble

# Test screen capture
./scripts/run_automation.sh --test-capture

# Run full automation
./scripts/run_automation.sh --password YOUR_PASSWORD --debug
```

**Options:** See `./scripts/run_automation.sh --help`

---

### `run_monitor.sh`

**Purpose**: Run HDMI capture stream monitor

**What it does:**
- Activates virtual environment
- Sets PYTHONPATH
- Runs `automation.monitor_stream` module

**Usage:**
```bash
# Basic streaming (5 FPS, raw mode)
./scripts/run_monitor.sh

# With detection enabled
./scripts/run_monitor.sh --detection-on

# Custom FPS
./scripts/run_monitor.sh --fps 10
```

**Keyboard controls** (while running):
- **Q/ESC** - Quit
- **D** - Toggle YOLO detection
- **S** - Save screenshot
- **F** - Toggle FPS counter
- **H** - Toggle help
- **+/-** - Adjust confidence

**Options:** See `./scripts/run_monitor.sh --help`

---

### `run_browser_assistant.sh`

**Purpose**: Run Interactive Browser Assistant

**What it does:**
- Activates virtual environment
- Sets PYTHONPATH
- Runs `automation.browser_assistant` module

**Usage:**
```bash
# Basic mode
./scripts/run_browser_assistant.sh

# Debug mode (recommended for first time)
./scripts/run_browser_assistant.sh --debug

# Custom video device
./scripts/run_browser_assistant.sh --device 1
```

**Chat Commands** (while running):
- **open chrome** - Launch Chrome browser
- **goto <url>** - Navigate to URL (e.g., `goto google.com`)
- **switch to doclayout** - Use DocLayout-YOLO model
- **switch to ui detection** - Use YOLOv11 UI detection model
- **analyze** - Analyze current screen
- **capture** - Save screenshot
- **help** - Show all commands
- **quit** - Exit assistant

**Options:** See `./scripts/run_browser_assistant.sh --help`

---

## Why These Scripts?

These helper scripts provide convenient wrappers around the Python modules:

1. **Environment setup** - Automatically activate venv and set PYTHONPATH
2. **Shorter commands** - `./scripts/run_monitor.sh` vs `source venv/bin/activate && export PYTHONPATH=... && python -m ...`
3. **Consistent behavior** - Always use the correct Python environment
4. **Error handling** - Check for venv existence before running

## Manual Usage

If you prefer to run Python modules directly:

```bash
# Activate venv
source venv/bin/activate

# Set PYTHONPATH
export PYTHONPATH=/path/to/ehr-ai-bridge-toolkit:$PYTHONPATH

# Run module
python -m automation.monitor_stream
python -m automation.windows_login
python -m automation.browser_assistant
```
