# Helper Scripts

This directory contains helper scripts for the EHR AI Bridge Toolkit automation tools.

## Scripts

### `setup_automation.sh`

**Purpose**: First-time setup for the automation environment

**What it does:**
- Creates Python virtual environment (`venv/`)
- Prefers Python 3.12/3.11 when available for MLX / EasyOCR compatibility
- Installs all dependencies from `requirements.txt`
- Creates `.env` file from template if it doesn't exist
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

### `run_history_panel_analyzer.sh`

**Purpose**: Compare OCR/layout strategies for the past-history panel on a saved screenshot

**What it does:**
- Activates virtual environment
- Sets `PYTHONPATH`
- Runs `automation.history_panel_analyzer`
- Compares EasyOCR full-image / EasyOCR + UI detection paths
- Infers a history ROI from date-like OCR anchors

**Usage:**
```bash
./scripts/run_history_panel_analyzer.sh captures/0410.jpg --date 20260410
```

**Output:** `automation_outputs/history_panel_analysis/<run-name>/`

---

---

### `start_mlx_vlm_server.sh`

**Purpose**: Start the local `mlx_vlm.server` process used by history matching and segmentation probes

**What it does:**
- Activates virtual environment
- Starts `mlx_vlm.server` on port **8181**
- Uses `mlx-community/Qwen3.5-4B-MLX-4bit` by default, or `gemma` / any model ID you pass
- Provides an OpenAI-compatible API (`/v1/chat/completions`) for local LLM inference on Apple Silicon
- `automation.mlx_vlm_history` sends **image + EasyOCR candidate list**
- `automation.mlx_vlm_segment_probe` still uses text-only prompts

**Usage:**
```bash
bash scripts/start_mlx_vlm_server.sh
bash scripts/start_mlx_vlm_server.sh qwen
bash scripts/start_mlx_vlm_server.sh gemma
bash scripts/start_mlx_vlm_server.sh mlx-community/Qwen3.5-4B-MLX-4bit
```

Keep this running in a separate terminal before using `automation.mlx_vlm_segment_probe` or `automation.mlx_vlm_history`.

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
- When the BLE connection drops unexpectedly, the server exits and `start_ble_server.sh` restarts it after about 3 seconds.
- `automation.ble_server` now watches the connection both via the BLE disconnect callback and via a periodic health check, so sleep/resume cases that miss the callback can still be recovered.
- Stop the server with **Ctrl+C** or SIGTERM to end the restart loop.

Example output on disconnection and recovery:
```
[2026-04-10 10:23:45] BLE デバイスが切断されました。
[2026-04-10 10:23:45] BLE 切断を検知。サーバーをシャットダウンします（start_ble_server.sh が再起動します）...
[2026-04-10 10:23:48] BLE サーバーが終了しました (exit code: 0)。3秒後に再起動します...
[2026-04-10 10:23:51] 接続成功: AA:BB:CC:DD:EE:FF
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
