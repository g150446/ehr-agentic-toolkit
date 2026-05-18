# EHR Agentic Toolkit

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**Python automation pipeline for on-premises EHR systems.**

This component provides HDMI capture, OCR, AI-assisted text input, and automated clinical document generation for on-premises Electronic Health Record systems.

Part of the [EHR Agentic Toolkit](../README.md).

## Key Features

- **Universal EHR Compatibility** - Works with any on-premises EHR via HDMI capture
- **Privacy-First Design** - All processing happens locally, patient identifiers never stored
- **AI-Assisted Clinical Support** - Summaries, differential diagnosis, treatment suggestions (planned)
- **Plugin Architecture** - Easy to add support for new EHR systems
- **Zero EHR Modification** - No changes to existing systems required
- **Real-time Stream Monitor** - Debug HDMI capture with YOLO detection visualization
- **ESP32 BLE Control** - Keyboard/mouse HID emulation over Bluetooth
- **GUI Image Analyzer** - Find text coordinates and textbox positions in screenshots
- **BLE Test CLI** - Interactive testing tool for ESP32 keyboard/mouse control
- **Discharge Summary Automation** (`ehr_composer`) - Reads past charts via scroll + OCR/VLM, generates structured summaries, and inputs them into Word documents automatically
- **Demo Video Recording** (`--movie`) - Records HDMI capture as phase-specific MP4s with fast-forward for hackathon demos

## Project Status

| Component | Status | Description |
|-----------|--------|-------------|
| **HDMI Capture** | Complete | Real-time video capture from MiraBox/compatible devices |
| **Layout Analysis** | In Progress | Evaluating ROI inference and detector-first OCR for EHR layout parsing |
| **OCR** | Complete | ndlocr-lite (DEIM+PARSEQ) for past chart reading; EasyOCR for Word UI detection |
| **Stream Monitor** | Complete | Interactive HDMI capture monitor with detection overlay |
| **ESP32 BLE Control** | Complete | Keyboard/mouse HID emulation over Bluetooth |
| **BLE Test CLI** | Complete | Interactive testing tool for ESP32 keyboard/mouse |
| **GUI Image Analyzer** | Complete | Text coordinate detection and textbox finding |
| **EHR Adapters** | In Progress | Fujitsu adapter framework implemented |
| **AI Engine** | In Progress | Gemma 4 26B integration for clinical support |
| **Clinical Decision Support** | Planned | Differential diagnosis, treatment suggestions |

**Current Focus:** Building automation infrastructure and screen capture pipeline.
**Next Steps:** EHR adapter development and AI engine integration.

## Architecture

```
+-----------------+
|  EHR System     |
|  (On-Premises)  |
+--------+--------+
         | HDMI
         ▼
+-----------------+
| Capture Layer   |◄── Screen Capture & OCR
+--------+--------+
         |
         ▼
+-----------------+
| Adapter Layer   |◄── EHR-specific Parsing
+--------+--------+
         |
         ▼
+-----------------+
| AI Engine       |◄── Gemma 4 26B / Local LLM
+--------+--------+
         |
         ▼
+-----------------+
| Clinical Output |◄── Decision Support
+-----------------+
```

## Quick Start

### Prerequisites

**Hardware:**
- macOS 11+ (M1 or later recommended) or Linux
- **For AI inference (Gemma 4 26B 4-bit): Apple Silicon M4+ with 24GB+ RAM** (runs via omlx or ollama)
- HDMI capture device (e.g., MiraBox, Elgato)
- **ESP32-S3 device** (e.g., M5AtomS3U) with wireless-input-bridge.ino flashed (for Windows automation via BLE HID)

**Software:**
- Python 3.10+

### Installation

**Option 1: Automated Setup (Recommended)**
```bash
# Clone the repository (including submodules)
git clone --recurse-submodules https://github.com/g150446/ehr-agentic-toolkit.git
cd ehr-agentic-toolkit

# Run setup script (installs everything)
./scripts/setup_automation.sh

# Edit configuration
cp .env.example .env
nano .env  # Configure your settings
```

**Option 2: Manual Setup**
```bash
# Clone and enter directory (including submodules)
git clone --recurse-submodules https://github.com/g150446/ehr-agentic-toolkit.git
cd ehr-agentic-toolkit

# Create virtual environment (Python 3.12 recommended)
python3.12 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install OCR dependencies
cd hardware-agent
./venv/bin/pip install onnxruntime

# Create .env file
cp .env.example .env
```

> **既存クローンに submodule を追加する場合:**
> ```bash
> git submodule update --init
> cd hardware-agent && ./venv/bin/pip install onnxruntime
> ```

## Usage

### HDMI Capture Stream Monitor

Real-time video monitoring tool with optional YOLO detection overlay.

**Use Cases:**
- Debug HDMI capture device connection
- Monitor Windows login screen during automation
- Visualize YOLO detection in real-time
- Capture screenshots for analysis

**Basic Usage:**
```bash
# Start monitor (5 FPS, raw video)
./scripts/run_monitor.sh

# Enable YOLO detection from start
./scripts/run_monitor.sh --detection-on

# Custom frame rate
./scripts/run_monitor.sh --fps 10

# Higher confidence threshold
./scripts/run_monitor.sh --confidence 0.3 --detection-on
```

**Interactive Controls (While Running):**
- **Q** or **ESC** - Quit
- **D** - Toggle YOLO detection ON/OFF
- **S** - Save screenshot
- **F** - Toggle FPS counter
- **H** - Toggle help overlay
- **+** / **-** - Adjust confidence threshold

**Output:** Screenshots saved to `monitor_outputs/`, logs in `automation_outputs/logs/`

---

### HDMI Snapshot Capture

Simple capture tool to save a single still image from the HDMI capture device.

```bash
# Save with a timestamped filename (captures/windows_capture_YYYYMMDD_HHMMSS.jpg)
python scripts/capture_windows.py

# Specify a filename (saved to captures/)
python scripts/capture_windows.py myshot.jpg

# Extension is auto-added if omitted
python scripts/capture_windows.py myshot
```

**Output:** All images are saved to the `captures/` directory.

---

### Past Chart Column OCR / Layout Comparison

Run **OCR anchor ROI estimation** and **OCR / layout strategy comparison** for past chart columns on saved images.

```bash
./scripts/run_history_panel_analyzer.sh captures/0410.jpg --date 20260410
```

This command compares:

- EasyOCR + full-screen OCR
- EasyOCR + UI detection OCR

Output is saved to `automation_outputs/history_panel_analysis/<run-name>/`.

---

### omlx VLM Server

`automation.mlx_vlm_history`, `automation.mlx_vlm_segmentation`, and `automation.mlx_vlm_ime` all use **omlx** (OpenAI-compatible API, port 8000). Start the omlx server beforehand.

```bash
# Check omlx server status
curl -s -H "Authorization: Bearer omlxkey" http://localhost:8000/v1/models
```

> **Known Issue:** `click_history` / `mlx_vlm_history` still has date misselection issues in the past chart column. This remains unresolved at this time.

### EHR Input with Text Files

`automation.ehr_input` accepts not only plain text but also **text file paths**. If a readable file is specified, its contents are sent via the remote keyboard using the existing Japanese/English/mixed input flow.

The current `ehr_input` uses **Gemma 4 26B** as the main model for Japanese segment splitting during long text input, with romaji corrected via a local dictionary while typing sequentially. IME candidate verification is also done primarily with Gemma 4 26B, avoiding blind Enter confirmation of unverified candidates. **Katakana segments are always extracted even within mixed segments and confirmed via F7 full-width katakana conversion**, so katakana parts of words like `アレルギー性` are not passed through kanji conversion candidates. Symbols that are difficult to handle over BLE are normalized to readable alternatives before input; for example, `℃` is sent as `C` and `×` as `x`. Furthermore, in helper reset, **the patient_record third pane coordinates are detected once at command start and validated by VLM, and the same crop coordinates are reused for subsequent Escape comparisons**. On the comparison string side, an **anchor tail** consisting of the last Japanese anchor concatenated with any immediately following confirmed ASCII/symbol suffix (e.g., `症状(`) is preserved, so even if that suffix remains after Escape, it is treated as a normal state.

Full-width symbols that should be displayed as-is are currently given special handling: **`、` `。` `・` `ー` `〜` `「` `」` `『` `』`**. `、` `。` `・` `〜` `「` `」` `『` `』` are **isolated as standalone tokens** from surrounding words and sent in Japanese mode, then immediately confirmed with **Enter**. **The long vowel mark `ー` is only instantly confirmed when it appears alone**; when it follows hiragana or katakana, it is kept as part of that word and converted/confirmed together (e.g., `コーテフ`, `えーと`, `アレルギー`). Other full-width symbols (e.g., `（` `）` `％` `：` `［` `］` `【` `】`) are currently **normalized to half-width ASCII** before sending. Multi-character replacements for medical context are also applied, such as `→` → `->`, `⇒` → `=>`, `℃` → `C`, `×` → `x`.

```bash
python -m automation.ehr_input data/patient_records/asthma_1.txt
python -m automation.ehr_input "open test" data/patient_records/asthma_1.txt
```

To show help:

```bash
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

### Discharge Summary Composer (`ehr_composer`)

Fully automated discharge summary pipeline that reads past charts, generates a structured summary via VLM, and inputs it into a Word document.

**Workflow:**
1. Scroll and read past chart entries via OCR + VLM
2. Generate a 7-section discharge summary (Chief Complaint, Present Illness, Past History, Hospital Course, Discharge Status, Discharge Plan, Discharge Prescriptions)
3. Open the discharge summary Word template from the EHR
4. Input each line via Notepad IME conversion, cut (`Ctrl+X`), switch to Word (`Alt+Tab`), and paste (`Ctrl+V`)

```bash
# Generate discharge summary (saves summary to logs/summary_YYYYMMDD_HHMMSS.txt)
python -m automation.ehr_composer --summary

# Skip chart reading — reload latest saved summary and input directly
python -m automation.ehr_composer --summary-no-scroll

# With demo video recording (--movie)
python -m automation.ehr_composer --summary --movie
```

**Summary Persistence**: Generated summaries are saved to `logs/summary_YYYYMMDD_HHMMSS.txt` after Phase 2. `--summary-no-scroll` loads the most recent saved summary, so you can re-run the input phase without repeating chart reading and VLM generation.

**OCR Backends**:
- Past chart reading: **ndlocr-lite** (DEIM detector + PARSEQ recognizer) by default. Switch to EasyOCR with `OCR_BACKEND=easyocr`.
- Word UI label detection ("担当医" etc.): **EasyOCR** (fixed — ndlocr-lite cannot detect small printed fonts in modern Word documents).
- IME popup candidates: **ndlocr-lite → EasyOCR → VLM** cascade.

```bash
# Use EasyOCR for past chart reading instead
OCR_BACKEND=easyocr python -m automation.ehr_composer --summary
```

**Demo Video (`--movie`)**: Records HDMI capture as phase-specific MP4s with fast-forward:
- Scroll phase: 3x speed
- UI interaction phase: Normal speed
- Text input phase: 2x speed
- VLM processing phase: Skipped (no screen changes)

Videos are saved to `captures/movie/`.

---

### BLE Test CLI

Use `automation.ble_test_cli` for standalone testing of the ESP32 BLE keyboard/mouse.

```bash
python -m automation.ble_test_cli
```

After connecting, you can confirm the Escape key with either of the following:

```text
press esc
esc
```

`press escape` is also normalized to `esc` and behaves the same.

---

### GUI Image Analyzer

Analyze screenshots to find text coordinates and textbox positions for GUI automation.

**Find Text Coordinates:**
```bash
# Find coordinates of specific text
python -m automation.gui_image_analyzer screenshot.png "患者検索"
# Output: Text "患者検索" found at coordinates: (x=80, y=462)
```

**Find Textbox Next to Label:**
```bash
# Find textbox to the right of a label
python -m automation.gui_image_analyzer screenshot.png --find-textbox "フリガナ"
# Output: Textbox right of "フリガナ" detected visually at: (x=333, y=684)

# Works for any form label
python -m automation.gui_image_analyzer form.png --find-textbox "氏名"
python -m automation.gui_image_analyzer form.png --find-textbox "生年月日"
```

**Use Cases:**
- Locate form fields before automated data entry
- Find button positions for click automation
- Analyze existing GUI layouts programmatically

---

### EHR Bridge (Production)

Full EHR integration with AI decision support (planned).

```bash
# Configure EHR system
ehr-bridge configure --ehr-type fujitsu

# Start the bridge
ehr-bridge start

# Run in interactive mode
ehr-bridge start --interactive

# Test EHR connection
ehr-bridge test-ehr

# View logs
ehr-bridge logs --tail 50
```

## Supported EHR Systems

| EHR System | Status | Adapter |
|------------|--------|---------|
| Fujitsu EHR | Supported | Built-in |
| NEC MegaOak HR | In Progress | Community |
| Philips Tasy | In Progress | Community |
| Medicom | Planned | - |
| Custom/Generic | Supported | Configuration-based |

Don't see your EHR? Create a custom adapter using our [adapter development guide](docs/custom-ehr-setup.md).

## Documentation

**User Guides:**
- [Getting Started Guide](docs/getting-started.md)
- [Automation Tools Guide](automation/README.md) - Start here for automation
- [Script Reference](scripts/README.md)

**Technical Documentation:**
- [Architecture Overview](docs/architecture.md)
- [EHR Configuration Guide](docs/ehr-configuration-guide.md)
- [API Reference](docs/api-reference.md)
- [Custom Adapter Development](docs/custom-ehr-setup.md)

## Troubleshooting

### Common Issues

#### "ModuleNotFoundError: No module named 'automation'"

**Solution:** Make sure you're using the helper scripts from the project root:
```bash
cd /path/to/ehr-agentic-toolkit
./scripts/run_monitor.sh
```

Or set PYTHONPATH manually:
```bash
export PYTHONPATH=/path/to/ehr-agentic-toolkit:$PYTHONPATH
python -m automation.monitor_stream
```

#### "No module named 'easyocr'"

**Solution:** Install EasyOCR in the active virtual environment:
```bash
source venv/bin/activate
python -m pip install easyocr
```

#### "No module named 'onnxruntime'"

**Solution:** Install onnxruntime in the active virtual environment:
```bash
source venv/bin/activate
pip install onnxruntime
```

#### ESP32 Not Connecting

**Checklist:**
1. Bluetooth enabled on Mac/PC
2. ESP32 powered and advertising
3. Device name in `.env` matches ESP32's advertised name
4. Test connection: `./scripts/run_ble_test.sh`

#### Screen Capture Not Working

**Device Detection:**
```bash
# List video devices (macOS)
system_profiler SPCameraDataType

# Test specific device index
./scripts/run_monitor.sh --device 1
```

#### OpenCV Window Not Appearing (macOS)

**Solution:** OpenCV requires GUI access. Run from terminal, not SSH:
```bash
# Run locally, not via SSH
./scripts/run_monitor.sh
```

If using remote connection, use VNC or enable X11 forwarding.

#### Virtual Environment Issues

**Solution:** Recreate the virtual environment:
```bash
# Remove old venv
rm -rf venv

# Run setup again
./scripts/setup_automation.sh
```

### Getting Help

- **Documentation:** Check [automation/README.md](automation/README.md) for detailed usage
- **Issues:** Report bugs at [GitHub Issues](https://github.com/g150446/ehr-agentic-toolkit/issues)
- **Logs:** Check `automation_outputs/logs/` for detailed error messages

## Security & Privacy

This toolkit is designed with healthcare privacy regulations in mind:

- **No PHI Storage**: Patient identifiable information is never persisted
- **Local Processing**: All AI processing can run entirely offline
- **Audit Logging**: Comprehensive activity logs for compliance

See [Security Guidelines](docs/security-guidelines.md) for detailed information.

## Development

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run tests with coverage
pytest --cov

# Format code
black ehr_ai_bridge/
ruff check ehr_ai_bridge/

# Type checking
mypy ehr_ai_bridge/
```

## Contributing

Contributions are welcome! Please read our [Contributing Guide](CONTRIBUTING.md) first.

### Adding Support for a New EHR

1. Create adapter in `ehr_ai_bridge/adapters/your_ehr/`
2. Implement `BaseEHRAdapter` interface
3. Add configuration YAML
4. Write tests
5. Submit PR with documentation


