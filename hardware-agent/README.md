# EHR Agentic Toolkit

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**Python automation pipeline for on-premise EHR systems that auto-generates and inputs discharge summaries.**

Captures the screen via HDMI, reads past medical records by scrolling with OCR + VLM, generates a 7-section discharge summary, and automatically inputs it into a Word document. All processing runs entirely locally; no patient data is transmitted externally.

Part of the [EHR Agentic Toolkit](../README.md).

## Key Features

- **Discharge Summary Automation** (`ehr_composer`) — Fully automated pipeline from past chart reading to discharge summary Word input
- **Privacy-First Design** — All processing runs entirely locally; no patient data is stored or transmitted
- **Universal EHR Compatibility** — Works with any on-premise EHR via HDMI capture
- **ESP32 BLE Control** — Bluetooth keyboard/mouse HID emulation
- **Zero EHR Modification** — No modification to existing systems required
- **Demo Video Recording** (`--movie`) — Phase-by-phase MP4 recording (for hackathon demos)

## Project Status

| Component | Status | Description |
|-----------|--------|-------------|
| **ehr_composer (Discharge Summary)** | Complete | Past chart reading → summary generation → Word auto-input |
| **HDMI Capture** | Complete | Real-time video capture from MiraBox and compatible devices |
| **OCR** | Complete | ndlocr-lite (DEIM+PARSEQ) for past chart reading; EasyOCR for Word UI detection |
| **ESP32 BLE Control** | Complete | Keyboard/mouse HID emulation over Bluetooth |
| **EHR Adapters** | In Progress | Fujitsu adapter framework implemented |
| **AI Engine** | In Progress | Gemma 4 26B integration for clinical support |
| **Clinical Decision Support** | Planned | Differential diagnosis, treatment suggestions |

## Quick Start

### Prerequisites

**Hardware:**
- macOS 11+ (Apple Silicon M4+ with 24GB+ RAM recommended for Gemma 4 26B 4-bit)
- HDMI capture device (e.g., MiraBox, Elgato)
- **ESP32-S3 device** (e.g., M5AtomS3U) with wireless-input-bridge.ino flashed (for Windows automation via BLE HID)

**Software:**
- Python 3.10+

### Installation

**Option 1: Automated Setup (Recommended)**
```bash
git clone --recurse-submodules https://github.com/g150446/ehr-agentic-toolkit.git
cd ehr-agentic-toolkit
./scripts/setup_automation.sh
cp .env.example .env
nano .env
```

**Option 2: Manual Setup**
```bash
git clone --recurse-submodules https://github.com/g150446/ehr-agentic-toolkit.git
cd ehr-agentic-toolkit
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cd hardware-agent
./venv/bin/pip install onnxruntime
cp .env.example .env
```

> **Adding submodules to an existing clone:**
> ```bash
> git submodule update --init
> cd hardware-agent && ./venv/bin/pip install onnxruntime
> ```

## Usage

### Discharge Summary Composer (`ehr_composer`)

Main command that fully automates discharge summary generation and input.

**Hardware setup required before running:**

1. **HDMI capture board** — Connect the Windows EHR PC (video source) to the AI Mac (capture host) via an HDMI capture card (e.g., MiraBox). The Mac reads the EHR screen through this link.
2. **M5Atom S3 BLE device** — Flash the M5Atom S3 (or compatible ESP32-S3) with `wireless-input-bridge.ino` and plug it into a USB port on the Windows EHR PC. It acts as a BLE keyboard/mouse, receiving keystrokes from the Mac and replaying them on the Windows machine.
3. **BLE server** — Start the BLE relay server on the Mac before running `ehr_composer`:
   ```bash
   ./scripts/start_ble_server.sh
   ```
   Keep this terminal running in the background throughout the session.

**Workflow:**
1. Scroll through past medical records and extract text with OCR + VLM
2. Generate a 7-section discharge summary (chief complaint, present illness, past history, hospital course, discharge status, discharge plan, discharge prescriptions)
3. Open the discharge summary Word template from the EHR
4. For each line: IME conversion in Notepad → cut (`Ctrl+X`) → paste into Word (`Ctrl+V`)

```bash
# Read past charts, generate discharge summary, and input into Word (main command)
python -m automation.ehr_composer --summary

# Skip chart reading and summary generation; re-use the previously saved summary for input only
python -m automation.ehr_composer --summary-no-scroll

# Also record demo video
python -m automation.ehr_composer --summary --movie
```

**Summary Persistence**: After running `--summary`, the generated summary is automatically saved to `logs/summary_YYYYMMDD_HHMMSS.txt`. `--summary-no-scroll` loads the latest saved summary, so you can re-run just the input phase.

**OCR Backends**:
- Past chart reading: **ndlocr-lite** (DEIM + PARSEQ) used by default. Switch with `OCR_BACKEND=easyocr`.
- Word UI label detection (e.g., "担当医"): **EasyOCR** (fixed — ndlocr-lite cannot detect small printed fonts in modern Word documents)
- IME popup candidates: **ndlocr-lite → EasyOCR → VLM** cascade

```bash
# Switch past chart reading to EasyOCR
OCR_BACKEND=easyocr python -m automation.ehr_composer --summary
```

**Demo Video (`--movie`)**: Saves per-phase MP4 files to `captures/movie/`.
- Scroll phase: 3× speed
- UI operation phase: 1× speed
- Text input phase: 2× speed
- VLM processing phase: skipped (no screen change)

---

## Debug / Development Tools

> The following tools are used for debugging and troubleshooting `ehr_composer` during development. Not needed for normal operation.

### omlx VLM Server (Prerequisite)

`automation.mlx_vlm_history`, `automation.mlx_vlm_segmentation`, and `automation.mlx_vlm_ime` all use **omlx** (OpenAI-compatible API on port 8000). Start the omlx server before running these tools.

```bash
# Check server status
curl -s -H "Authorization: Bearer omlxkey" http://localhost:8000/v1/models
```

> **Known Issue:** `click_history` / `mlx_vlm_history` still has occasional incorrect date selection in the past chart column.

---

### EHR Input (`ehr_input`)

Debug tool for inputting text and Japanese files via BLE keyboard. Used for unit testing the input phase of `ehr_composer`.

The current `ehr_input` uses **Gemma 4 26B** as the main model for Japanese segment splitting during long text input, with romaji corrected via a local dictionary while typing sequentially. IME candidate verification is also done primarily with Gemma 4 26B, avoiding blind Enter confirmation of unverified candidates. **Katakana segments are always extracted even within mixed segments and confirmed via F7 full-width katakana conversion**, so katakana parts of words like `アレルギー性` are not passed through kanji conversion candidates. Symbols that are difficult to handle over BLE are normalized to readable alternatives before input; for example, `℃` is sent as `C` and `×` as `x`. Furthermore, in helper reset, **the patient_record third pane coordinates are detected once at command start and validated by VLM, and the same crop coordinates are reused for subsequent Escape comparisons**. On the comparison string side, an **anchor tail** consisting of the last Japanese anchor concatenated with any immediately following confirmed ASCII/symbol suffix (e.g., `症状(`) is preserved, so even if that suffix remains after Escape, it is treated as a normal state.

Full-width symbols that should be displayed as-is are currently given special handling: **`、` `。` `・` `ー` `〜` `「` `」` `『` `』`**. `、` `。` `・` `〜` `「` `」` `『` `』` are **isolated as standalone tokens** from surrounding words and sent in Japanese mode, then immediately confirmed with **Enter**. **The long vowel mark `ー` is only instantly confirmed when it appears alone**; when it follows hiragana or katakana, it is kept as part of that word and converted/confirmed together (e.g., `コーテフ`, `えーと`, `アレルギー`). Other full-width symbols (e.g., `（` `）` `％` `：` `［` `］` `【` `】`) are currently **normalized to half-width ASCII** before sending. Multi-character replacements for medical context are also applied, such as `→` → `->`, `⇒` → `=>`, `℃` → `C`, `×` → `x`.

```bash
python -m automation.ehr_input data/patient_records/asthma_1.txt
python -m automation.ehr_input "open test" data/patient_records/asthma_1.txt
python -m automation.ehr_input --help
```

#### IME Mode Detection

During input, a single `a` is typed and a screen capture is passed to the VLM to determine whether `a` (English mode) or `あ` (Japanese mode) is displayed. Cleanup sends Backspace in English mode, and in Japanese mode sends Escape followed by **Backspace only if uncommitted composition remains**, so previously confirmed characters are not destroyed.

| Option | Description |
|---|---|
| `--clear` | Send Backspace 50 times to clear the field before input |
| `--fireworks <model>` | Switch segmentation, IME candidate reading, and helper word suggestions to Fireworks AI model |
| `--google-ai-studio` | Switch segmentation, IME candidate reading, and helper word suggestions to Google AI Studio `gemma-4-26b-a4b-it` |
| `--novita [model]` | Switch segmentation, IME candidate reading, and helper word suggestions to Novita AI. Defaults to `google/gemma-4-26b-it` if model is omitted |
| `--openrouter [model]` | Switch segmentation, IME candidate reading, and helper word suggestions to OpenRouter vision-capable model. Model specification required for standalone use |
| `--openrouter --novita [model]` | Alternate between OpenRouter and Novita for each eligible VLM request. Both use the same model ID, defaulting to `google/gemma-4-26b-it` if omitted |
| `--mactest` | Use Mac local display + `pyautogui` instead of HDMI/BLE for testing |

#### Maintenance Notes

- Logs for each run are saved to `logs/*.txt`, with the **executed filename, raw command line, and parsed option summary** recorded at the top.
- If conversion is not as expected, follow the `[VLM match]`, `[candidate check/romaji]`, `[attempt N]`, `[helper word]` lines in order to easily distinguish whether the issue is **wrong candidate number selection / candidate not found / fell back to subsequent fallback**.
- For pure kanji targets, the implementation **does not immediately adopt mixed candidates that only match the reading**, sending ambiguous candidates to helper word fallback or highlight candidate confirmation instead.
- Just before entering helper word fallback, the screen is re-captured after each `Escape`, and the VLM compares the **baseline image saved at the last successful confirmation** with the **current image after Esc**. For patient_record, **the third pane coordinates are detected once at command start and validated by VLM**, and subsequent compare crops reuse those fixed coordinates. **If Windows Notepad is detected**, the traditional Notepad body region is cropped to exclude the top menu bar and Windows taskbar. The comparison uses the **last confirmed string** as the baseline, and if there is a confirmed ASCII/symbol suffix immediately after the Japanese anchor, it is included in the anchor tail (e.g., `症状(`). This allows cases where `症状(` remains at the normal end while only the uncommitted part disappears to be correctly treated as reset complete. `captures/` retains `debug_panel_detection_*`, `debug_helper_reset_*_compare_crop.png`, and `debug_vlm_input_helper_reset_compare_*`, and the `[helper reset][compare]` lines in `logs/*.txt` show the yes/no decision after each `Esc`.

> **Known Issue:** In re-validation with `data/patient_records/asthma_1.txt`, the blank-stall issue has been resolved, but misconversion still remains for words like `咽頭痛`. Especially at the beginning of long texts, unreflected symbols like `[` and conversion fluctuation around `昨晩` / `咳嗽` remain.

---

### HDMI Capture Stream Monitor

Debug tool for verifying HDMI capture device connection and real-time YOLO detection visualization.

```bash
./scripts/run_monitor.sh
./scripts/run_monitor.sh --detection-on
./scripts/run_monitor.sh --fps 10
./scripts/run_monitor.sh --confidence 0.3 --detection-on
```

**Interactive Controls:** Q/ESC — Quit, D — Toggle YOLO, S — Screenshot, F — FPS counter, H — Help, +/- — Confidence

**Output:** Screenshots saved to `monitor_outputs/`, logs in `automation_outputs/logs/`

---

### HDMI Snapshot Capture

Saves a single still image from the HDMI capture device.

```bash
python scripts/capture_windows.py
python scripts/capture_windows.py myshot.jpg
```

**Output:** All images are saved to the `captures/` directory.

---

### Past Chart Column OCR / Layout Comparison

Debug tool for comparative analysis of OCR strategies on saved images.

```bash
./scripts/run_history_panel_analyzer.sh captures/0410.jpg --date 20260410
```

Output is saved to `automation_outputs/history_panel_analysis/<run-name>/`.

---

### BLE Test CLI

Unit testing tool for ESP32 BLE keyboard/mouse HID.

```bash
python -m automation.ble_test_cli
```

---

### GUI Image Analyzer

Debug tool for detecting text coordinates and textbox positions from screenshots.

```bash
python -m automation.gui_image_analyzer screenshot.png "患者検索"
python -m automation.gui_image_analyzer screenshot.png --find-textbox "フリガナ"
```

---

### EHR Bridge (Production)

Full EHR integration with AI decision support (planned).

```bash
ehr-bridge configure --ehr-type fujitsu
ehr-bridge start
```

## Troubleshooting

### Common Issues

#### "ModuleNotFoundError: No module named 'automation'"

```bash
export PYTHONPATH=/path/to/ehr-agentic-toolkit:$PYTHONPATH
python -m automation.ehr_composer --summary
```

#### "No module named 'easyocr'"

```bash
source venv/bin/activate
python -m pip install easyocr
```

#### "No module named 'onnxruntime'"

```bash
source venv/bin/activate
pip install onnxruntime
```

#### ESP32 Not Connecting

1. Bluetooth enabled on Mac/PC
2. ESP32 powered and advertising
3. Device name in `.env` matches ESP32's advertised name
4. Test connection: `./scripts/run_ble_test.sh`

#### Screen Capture Not Working

```bash
system_profiler SPCameraDataType
./scripts/run_monitor.sh --device 1
```

#### Virtual Environment Issues

```bash
rm -rf venv
./scripts/setup_automation.sh
```

### Getting Help

- **Issues:** Report bugs at [GitHub Issues](https://github.com/g150446/ehr-agentic-toolkit/issues)
- **Logs:** Check `logs/` for detailed run logs

## Security & Privacy

- **No PHI Storage**: Patient identifiable information is never persisted
- **Local Processing**: All AI processing runs entirely offline

## Contributing

Contributions are welcome! Please read our [Contributing Guide](CONTRIBUTING.md) first.
