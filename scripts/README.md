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
export PYTHONPATH=/path/to/ehr-agentic-toolkit:$PYTHONPATH

# Run module
python -m automation.monitor_stream
```
