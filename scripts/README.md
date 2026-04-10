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

---

### `start_mlx_vlm_server.sh`

**Purpose**: Start mlx_vlm inference server for Japanese text segmentation

**What it does:**
- Activates virtual environment
- Starts `mlx_vlm.server` on port **8181** with `mlx-community/gemma-4-e2b-it-4bit`
- Provides an OpenAI-compatible API (`/v1/chat/completions`) for local LLM inference on Apple Silicon

**Usage:**
```bash
bash scripts/start_mlx_vlm_server.sh
```

Keep this running in a separate terminal before using `automation.mlx_vlm_segment_probe`.

**Verify server is up:**
```bash
curl -s http://127.0.0.1:8181/v1/models
```

---

### `start_ble_server.sh`

**Purpose**: Start the resident BLE server that manages the ESP32 connection

**What it does:**
- Activates virtual environment
- Sets PYTHONPATH
- Runs `automation.ble_server` — a long-running process that connects to the ESP32 and listens on `/tmp/ble_server.sock`

**Usage:**
```bash
./scripts/start_ble_server.sh
```

Keep this running in a separate terminal before executing `ehr_input.py` or any other client that uses `BLEClient`.

**Auto-reconnect behavior:**
- When the BLE connection drops unexpectedly, the server logs the disconnection with a timestamp and waits 60 seconds before retrying.
- If reconnection fails, it retries again every 60 seconds until successful or until the server is stopped.
- Stop the server with **Ctrl+C** or SIGTERM; the server reconnects gracefully without losing the socket.

Example output on disconnection and recovery:
```
[2026-04-10 10:23:45] BLE デバイスが切断されました。
[2026-04-10 10:23:45] BLE 切断を検知。60秒後に再接続を試みます...
[2026-04-10 10:24:45] 再接続中...
[2026-04-10 10:24:52] 再接続成功: AA:BB:CC:DD:EE:FF
```

---

### `run_ble_test.sh`

**Purpose**: Run interactive BLE test CLI for ESP32 keyboard/mouse control

**Usage:**
```bash
./scripts/run_ble_test.sh
```

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
