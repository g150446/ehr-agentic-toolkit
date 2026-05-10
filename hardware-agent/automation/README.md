# Automation Tools

This directory contains automation tools for HDMI capture, screen analysis, and ESP32 BLE control.

## Tools

1. **HDMI Capture Stream Monitor** - Standalone real-time video monitoring with optional YOLO detection
2. **GUI Image Analyzer** - Find text coordinates and textbox positions in screenshots
3. **BLE Test CLI** - Interactive testing tool for ESP32 BLE keyboard and mouse

---

## EHR Input Automation

Automated EHR field input and patient chart opening via HDMI screen capture, OCR, and BLE mouse/keyboard control.

> **Prerequisites**: `ble_server.py` must be running before executing any of these functions.
> Start it with `./scripts/start_ble_server.sh` in a separate terminal.
> Before using `click_history` / `mlx_vlm_history`, ensure the omlx VLM server (http://localhost:8000) is running.

### Command Line Usage

`automation.ehr_input` switches behavior based on command line arguments.

> `--openrouter [model]` switches **segmentation, IME mode detection, candidate reading, and helper word suggestions** to OpenRouter. Since it also performs image-based IME reading, please specify a **vision-capable model**. Model specification is required for standalone use.
> `--fireworks <model>` switches the same external VLM calls to **Fireworks AI**'s OpenAI-compatible API (requires `FIREWORKS_API_KEY`).
> `--novita [model]` switches the same external VLM calls to **Novita AI**'s OpenAI-compatible API (requires `NOVITA_API_KEY`). Defaults to `google/gemma-4-26b-it` if model is omitted.
> `--openrouter --novita [model]` alternates between **OpenRouter and Novita per request**. Both use the same model ID, defaulting to `google/gemma-4-26b-it` if omitted.
> `--google-ai-studio` switches the same external VLM calls to **Google AI Studio's `gemma-4-26b-a4b-it`** (requires `GEMINI_API_KEY`).

```bash
# No arguments: open test patient chart
python -m automation.ehr_input

# Show help
python -m automation.ehr_input help
python -m automation.ehr_input --help

# Japanese text: IME conversion only
python -m automation.ehr_input 肺炎

# English text: direct input in alphanumeric mode
python -m automation.ehr_input tesuto

# Text file: read contents and input
python -m automation.ehr_input data/patient_records/asthma_1.txt

# Clear field before input (Backspace x50)
python -m automation.ehr_input --clear 肺炎

# Mixed Japanese-English text: auto-switch IME mode per segment
python -m automation.ehr_input "COVID-19の感染を確認した"

# Run segmentation, candidate reading, and helper word suggestions with OpenRouter model
python -m automation.ehr_input --openrouter google/gemma-4-26b-a4b-it "両肺野に"

# Run with Novita AI default model
python -m automation.ehr_input --novita "両肺野に"

# Run with specified Novita AI model
python -m automation.ehr_input --novita deepseek/deepseek-vl2 "聴診"

# Alternate between OpenRouter and Novita to distribute rate limits
python -m automation.ehr_input --openrouter --novita "両肺野に"

# Use the same shared model for both OpenRouter and Novita
python -m automation.ehr_input --openrouter --novita deepseek/deepseek-vl2 "聴診"

# Run with Fireworks AI model
python -m automation.ehr_input --fireworks accounts/fireworks/models/gemma-4-26b-a4b-it "両肺野に"

# Run segmentation, candidate reading, and helper word suggestions with Google AI Studio Gemma 4 26B A4B
python -m automation.ehr_input --google-ai-studio "両肺野に"

# Click cancel [F9] button to close chart
python -m automation.ehr_input "close record"

# First argument "open test", second argument is text: open chart then input
python -m automation.ehr_input "open test" 肺炎
python -m automation.ehr_input "open test" "MRI所見"
python -m automation.ehr_input "open test" data/patient_records/asthma_1.txt
```

When given Japanese text, `ehr_input.py` uses **Gemma 4 26B** as the main model for segment splitting, corrects romaji via a local dictionary, and then inputs through IME. If Gemma's splitting is too fine-grained and destabilizes IME candidates, it automatically falls back to `sudachipy + pykakasi` local segmentation. If the argument is a readable text file, its **contents** flow into the same input pipeline. In helper reset, **the patient_record third pane coordinates are detected once at command start and validated by VLM, and the same coordinates are reused for subsequent compare crops**. On the comparison string side, an **anchor tail** consisting of the last Japanese anchor concatenated with any immediately following confirmed ASCII/symbol suffix (e.g., `症状(`) is used for baseline/compare, so even if that suffix remains after Escape, it is judged as a normal reset completion.

Full-width symbols that should be displayed as-is are currently given special handling: **`、` `。` `・` `ー` `〜` `「` `」` `『` `』`**. `、` `。` `・` `〜` `「` `」` `『` `』` are **isolated as standalone tokens** from surrounding words and sent in Japanese mode, then immediately confirmed with **Enter**. **The long vowel mark `ー` is only instantly confirmed when it appears alone**; when it follows hiragana or katakana, it is kept as part of that word and converted/confirmed together (e.g., `コーテフ`, `えーと`, `アレルギー`). Other full-width symbols (e.g., `（` `）` `％` `：` `［` `］` `【` `】`) are currently **normalized to half-width ASCII** before sending.

Logs for each run are saved to `logs/*.txt`, with the **executed filename, raw command line, and parsed option summary** recorded at the top.

When checking conversion results during operation, following the `[VLM match]`, `[candidate check/romaji]`, `[attempt N]`, `[helper word]` lines makes it easy to distinguish whether the issue is wrong candidate number selection, candidate not found, or fallback trigger. For pure kanji targets, the implementation does not immediately adopt mixed candidates that only match the reading, sending ambiguous candidates to helper word fallback or highlight candidate confirmation instead. Also, if a romaji-matching candidate is shorter than the target, it is treated as a different word and not adopted (prevents e.g. selecting "明らかな" for "昭かな"). In helper reset compare `[helper reset][compare]`, judgment is based on the **initially fixed compare crop** and the **last confirmed string**, so if an anchor tail like `症状(` remains, it is not treated as extra characters.

**Kana-Kanji Cross-Type Guard**: In fuzzy matching (`_ime_candidate_matches`) when allowing 1-character differences, if the difference is a cross-type between kana (U+3040-30FF) and kanji (U+4E00-9FFF), it is rejected immediately as OCR noise rather than being accepted (prevents e.g. "直地に" → "直ちに" mismatches). Differences between kanji characters (e.g. 著名な→著明な) are only allowed if 3 or more characters differ.

**2-Character Fuzzy Disable**: For 2-character targets, fuzzy matching (1-character difference allowed) is disabled and only exact matches are accepted. For 2 characters, allowing 50% mismatch carries high risk of adopting a different word (prevents e.g. selecting "血症" for "血競" when it is actually "血漿"). VLM-assisted reading is triggered instead, enabling more accurate candidate recognition.

**Visual Similarity Leading Character (Pass 5)**: When accepting candidates that differ only in the leading character due to suffix matching, adoption is limited to cases where **both of the first 2 characters are kanji** (e.g. 署明な→著明な). Verb conjugations with kana suffixes (e.g. selecting "伴って" for "燈って") and non-kanji leading characters are not treated as visually similar.

**Medical Term Romaji Override**: The `_ROMAJI_OVERRIDES` dictionary in `_kanji_to_romaji()` manually overrides romaji for medical terms that pykakasi misreads (e.g. 生食→seishoku, 静注→seichuu). Invalid overrides are also prevented by `_validate_vlm_romaji`.

**Decomposition Typing Strategy**: Medical abbreviations not in the IME dictionary (e.g. `静注`, `筋注`) are input using carrier words that do exist in the dictionary, then deleting unwanted trailing characters with Backspace. Words registered in `_DECOMPOSE_OVERRIDES` are processed by `_type_kanji_via_decomposition()`. Example: `静注` → convert and confirm `静脈`(seimyaku) → BS×1 to delete `脈` → convert and confirm `注射`(chuusha) → BS×1 to delete `射` → result: `静注`. Carrier words are converted with `_strict=True`, so hiragana fallback does not occur. If a step fails, already committed characters are rolled back.

Sentences longer than 4 characters or containing particles are split into segments by `type_japanese_sentence()` and **input sequentially**. Among full-width symbols to display, `、` → `,` + Enter / `。` → `.` + Enter / `・` → `/` + Enter / `〜` → `~` + Enter / `「` `」` `『` `』` → Japanese mode bracket + Enter are confirmed independently. On the other hand, **`ー` is only instantly confirmed with `-` + Enter when it is a standalone token**; when it follows hiragana or katakana, it is kept within the word. Newlines and `[` `]` `(` `)` `%` `:` are also switched to dedicated key sends.

For mixed Japanese-English text (e.g. `"COVID-19の感染を確認した"`), ASCII-only segments are input in alphanumeric mode, and Japanese segments are input in hiragana mode, with IME automatically toggled.

> **Known Issue**: In re-validation with `data/patient_records/asthma_1.txt`, the blank-stall issue has been resolved, but misconversion remains for words like `咽頭痛` / `昨晩` / `咳嗽`. On actual screens, input has been confirmed to proceed to the beginning of the body text, but full automation remains unresolved.

### Local Segmentation Probe (Recommended)

A tool to verify the behavior of local Japanese segmentation using **sudachipy + pykakasi**. No external services required; long vowels are also accurate.

```bash
python -m automation.local_segment_probe "肺炎に対して、抗菌薬による治療を行う。"
```

Example output:

```
Target: '肺炎に対して、抗菌薬による治療を行う。'
Engine: sudachipy (SplitMode.C) + pykakasi (hepburn)
Segmentation summary: 肺炎(haien) / に(ni) / 対して(taishite) / 、(,) / 抗菌薬(koukinyaku) / に(ni) / よる(yoru) / 治療(chiryou) / を(wo) / 行う(okonau) / 。(.)
Segments:
  1. '肺炎' (haien)
  2. 'に' (ni)
  3. '対して' (taishite)
  4. '、' (,)
  5. '抗菌薬' (koukinyaku)
  ...
```

### mlx_vlm Segmentation Probe (Reference Implementation)

An LLM-based implementation using the omlx VLM server model. It is not used by `ehr_input.py` but can be used for LLM output comparison, etc. The omlx VLM server (http://localhost:8000) must be running. Despite the `mlx_vlm` name, this usage sends only text to `/v1/chat/completions` without images.

```bash
python -m automation.mlx_vlm_segment_probe "肺炎に対して抗菌薬による治療を行う"
```

### Ollama Segmentation Probe (Reference Implementation)

> **Ollama support has been removed.** Use mlx_vlm instead.

```bash
python -m automation.mlx_vlm_segment_probe "肺炎に対して抗菌薬による治療を行う"
```

### click_history

Detect and click the entry for a specified date in the past chart column.

1. Capture HDMI screen and run OCR
2. Extract date candidate coordinates with EasyOCR, then pass the **past chart column image + target date** to the omlx VLM server to read the visible date list from top to bottom
   - Fast path: VLM is unnecessary if uniquely identifiable by regex
   - The **rank of the target date** returned by the VLM is mapped to the vertical order of EasyOCR candidates to determine click coordinates
   - Uses the same recognition algorithm as `mlx_vlm_history.py`: **read date list from image / estimate coordinates with EasyOCR**
3. Move BLE mouse to detected coordinates and click

> **Prerequisite:** The omlx VLM server (http://localhost:8000) must be running.

> **Known Issue:** `click_history` / `mlx_vlm_history` still has date misselection issues in the past chart column. This remains unresolved at this time.

```bash
python -m automation.ehr_input "click history 20260312"
```

```python
from automation.ehr_input import click_history
click_history("20260312")
```

VLM connection destination, model, and timeout can be changed via environment variables:

| Environment Variable | Default | Description |
|---------|-----------|------|
| `MLX_VLM_HISTORY_URL` | `http://localhost:8000/v1/chat/completions` | omlx VLM server chat completion endpoint |
| `MLX_VLM_HISTORY_MODEL` | `gemma-4-26b-it` | Model to use |
| `MLX_VLM_HISTORY_TIMEOUT` | `120` | Timeout in seconds |

### edit_history

Click the entry for a specified date in the past chart column, wait 1 second, then click the edit button to enter edit mode.

1. Call `click_history(date_str)` to click the target date entry
2. Wait 1 second (for edit button to appear)
3. Re-capture HDMI screen
4. Detect edit button via OpenCV template matching (`match_templates/edit_button.jpg`)
5. Move BLE mouse to detected coordinates and click

```bash
python -m automation.ehr_input "edit history 20260312"
```

```python
from automation.ehr_input import edit_history
edit_history("20260312")
```

> **Template Image:** `match_templates/edit_button.jpg` must contain a cropped image of the edit button.
> If the matching score is below 0.7, a `RuntimeError` is raised.

#### Template Matching Verification (No Click)

The `mlx_vlm_history.py` CLI creates candidate positions with **full-image EasyOCR** on a saved image, then has the omlx VLM server **read the date list in the past chart column from top to bottom**. It maps that rank to EasyOCR coordinates to identify the date position. `click_history()` uses the same recognition algorithm. After that, edit button template matching is also performed and the coordinates are displayed (no click).

```bash
python -m automation.mlx_vlm_history captures/history.jpg 20260312
# → Displays date coordinates + edit button coordinates and matching score
```

### Past Chart Column Analysis

For saved images, **estimate the past chart column ROI with OCR anchors** while comparing EasyOCR full-image / EasyOCR + UI detection.

```bash
# Compare EasyOCR full-image / UI detection
python -m automation.history_panel_analyzer captures/0410.jpg --date 20260410

# Run from helper script
./scripts/run_history_panel_analyzer.sh captures/0410.jpg --date 20260410
```

By default, the following are compared:

1. `EasyOCR + full-image OCR`
2. `EasyOCR + UI detection OCR`

Output is saved to `automation_outputs/history_panel_analysis/<run-name>/`. Main artifacts:

- `*_annotated.png`: Candidate date boxes and estimated ROI
- `*_summary.txt`: OCR segment count, date candidate count, and matching candidates per strategy
- `history_roi.png`: Estimated past chart column ROI
- `summary.txt`: Overall comparison summary and recommendations
- `manifest.json`: Machine-readable comparison results

### close_record

Capture the HDMI screen, detect the "Cancel [F9]" button in the upper right via OCR, and click it to close the open chart.

1. Capture HDMI screen and search for "Cancel" text via OCR
2. Move BLE mouse to detected coordinates and click

```python
from automation.ehr_input import close_record
close_record()
```

### open_test_patient_chart

Automatically open the test patient's chart. Executes the following steps:

0. Capture HDMI screen, detect and click the "Patient Search" tab via OCR (wait 0.5 seconds). If the tab is already selected and appears in blue text making OCR detection impossible, skip detection and proceed to the next step
1. Search for and click the Furigana field via OCR, input `tesuto`, then press Enter to display the patient list
2. Wait 0.5 seconds then press Enter to select the first patient and open the chart
3. Wait 2 seconds then press Enter to close the dialog shown immediately after display

```python
from automation.ehr_input import open_test_patient_chart
open_test_patient_chart()
```

### type_kanji_via_ime

Type romaji into IME, then verify candidates from the HDMI capture crop using **VLM (Gemma 4 26B) preferred / OCR fallback** before confirming with Enter. Unverified candidates are never blindly confirmed with Enter.

**IME Candidate Detection Method**: On the patient-record screen, the candidate popup is first extracted using **PP-StructureV3** within the third panel; if that fails, it falls back to the traditional OpenCV-based method (blue selection bar / inverted region). PP-StructureV3 may automatically download the official model on first run.

**VLM Prompt**: To reduce misreading within the candidate window, the target string is explicitly placed at the beginning of the prompt with the instruction "Please read only the text of the line that is highlighted in black (inverted)."

```python
from automation.ehr_input import type_kanji_via_ime

# Type "haien", confirm "肺炎" in IME candidate, and confirm with Enter
type_kanji_via_ime("haien", "肺炎")

# Automatically convert kanji to romaji before execution
from automation.ehr_input import _kanji_to_romaji
romaji = _kanji_to_romaji("肺炎")  # → "haien"
type_kanji_via_ime(romaji, "肺炎")
```

### detect_ime_mode / ensure_ime_mode

Determine the current Windows IME input mode from a screen capture, and toggle if necessary.

**`detect_ime_mode(client, config)`**: Type a single `'a'`, then read the screen with the VLM (omlx VLM server) to detect the IME mode. English input mode shows `'a'`, while Japanese (hiragana) input mode shows `'あ'`. After determination, the typed character is deleted with Backspace. When `--openrouter` is specified, uses the OpenRouter model; when `--novita` is specified, uses the specified Novita AI model (default `google/gemma-4-26b-it` if omitted); when `--openrouter --novita [model]` is specified, alternates between OpenRouter and Novita using the shared model; when `--fireworks` is specified, uses the specified Fireworks AI model; when `--google-ai-studio` is specified, uses Google AI Studio's `gemma-4-26b-a4b-it`.

**`ensure_ime_mode(target_mode, client, current_mode)`**: Only sends `key:zenkaku` (half-width/full-width key) to toggle if the current mode differs from the target, and returns the new mode string. Designed without screen re-capture; the caller tracks the mode.

```python
from automation.ehr_input import detect_ime_mode, ensure_ime_mode
from automation.ble_client import BLEClient

client = BLEClient()
current = detect_ime_mode(client)                          # 'japanese' / 'english' / None
current = ensure_ime_mode("english", client, current)  # Send half-width/full-width if needed
```

### input_text_to_field

Low-level function to detect a labeled input field via OCR and input text.

```python
from automation.ehr_input import input_text_to_field

# Input "tesuto" into the Furigana field
input_text_to_field(input_text="tesuto", label="フリガナ")
```

---

## EHR Composer (Discharge Summary Automation)

`automation.ehr_composer` combines **past chart reading**, **VLM-based summary generation**, and **Word document input** into a single automated pipeline.

### What it does

1. **Phase 1 – Scroll & Read Past Charts**: Scrolls the past chart column while reading entries via OCR + VLM, merging results into a structured JSON array.
2. **Phase 2 – Generate Discharge Summary**: Sends the extracted chart data to the VLM and generates a discharge summary with 7 sections:
   - Chief Complaint, Present Illness, Past History, Hospital Course, Discharge Status, Discharge Plan, Discharge Prescriptions
3. **Phase 3 – Open Word + Notepad**: Clicks the discharge summary icon in the EHR, opens the Word template, then launches Notepad for IME conversion.
4. **Phase 4 – Line-by-Line Input**: Types each summary line into Notepad, cuts it (`Ctrl+X`), switches to Word (`Alt+Tab`), pastes (`Ctrl+V`), then returns to Notepad.

### Command Line Usage

```bash
# Basic usage (omlx is auto-enabled if omitted)
python -m automation.ehr_composer --summary

# Specify VLM model
python -m automation.ehr_composer --summary --omlx gemma-4-26b-a4b-it-4bit

# Record demo video (--movie)
python -m automation.ehr_composer --summary --movie
```

### `--movie` Option (Demo Video Recording)

When `--movie` is specified, the tool records the HDMI capture as phase-specific MP4 files saved to `captures/movie/`:

| Phase | Speed | Output File |
|-------|-------|-------------|
| Phase 1 (Scroll & Read) | **3x fast-forward** | `composer_scroll_<timestamp>.mp4` |
| Phase 2 (Summary Generation) | **Skipped** (no screen changes) | — |
| Phase 3 (Word/Notepad Launch) | **Normal speed** | `composer_other_<timestamp>.mp4` |
| Phase 4 (Text Input) | **2x fast-forward** | `composer_input_<timestamp>.mp4` |

The recording uses **monkey-patching** on `capture_screen()` so no changes are needed in existing modules (`ehr_reader.py`, `ehr_input.py`, `screen_analyzer.py`).

> **Note**: Requires `ffmpeg`-compatible codecs. OpenCV is used with fourcc fallback (`mp4v` → `avc1` → `XVID`).

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

### Features

- **Interactive REPL interface** - Natural command-line interaction with history
- **Colored output** - Visual feedback with success/error/info messages
- **Connection status tracking** - Always know if you're connected and in what mode
- **Comprehensive help** - Built-in help for all commands
- **Tab completion** - Command history with up/down arrows
- **Raw command mode** - Send any BLE UART command for debugging

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
- On macOS, also confirm your terminal app is allowed under **Settings > Privacy & Security > Bluetooth**
- Verify device name matches `ESP32_DEVICE_NAME` in `.env`
- Try the `scan` command to see available devices

**HDMI capture fails from SSH (`not authorized to capture video`)**
- `cv2.VideoCapture(...)` requires macOS **Camera** permission, not **Screen Recording**
- Run HDMI capture processes in a local GUI session with Camera permission granted

**"Not connected to BLE device"**
- Run `connect` command first
- Check that ESP32 is not connected to another device
- Try disconnecting and reconnecting

**"Failed to send command"**
- Verify you're in the correct mode (keyboard/mouse)
- Check ESP32 is still connected (`status` command)
- Try reconnecting

---

## GUI Image Analyzer

Analyze screenshots to find text coordinates and textbox positions for GUI automation.

### Quick Start

```bash
# Find text coordinates
python -m automation.gui_image_analyzer screenshot.png "患者検索"

# Find textbox right to a label
python -m automation.gui_image_analyzer screenshot.png --find-textbox "フリガナ"
```

### Features

- **Text Search** - Find coordinates of any text in an image
- **Textbox Detection** - Locate input fields next to labels using OCR + edge detection
- **Visual Fallback** - Detects empty textboxes visually when OCR finds no text
- **Japanese Support** - Works with Japanese and English text
- **OCR** - EasyOCR as default backend
- **YOLO UI Detection** - Detects individual UI elements (buttons, tabs, inputs) before OCR to prevent menu items from being merged into one text segment

### How It Works

**YOLO mode (default):**
1. **UI Element Detection** - `foduucom/web-form-ui-field-detection` (YOLOv8) detects individual UI elements
2. **Per-element OCR** - Each detected element is cropped and OCR'd separately
3. **Label Matching** - Finds the label text among individually recognized elements
4. **Textbox Search** - Looks for element to the right within vertical tolerance
5. **Coordinate Output** - Returns center (x, y) coordinates

**OCR mode (`--detection-mode ocr`):**
1. EasyOCR runs on the full image at once
2. Adjacent UI elements (e.g. horizontal menu tabs) may be merged into one text segment

Use `--detection-mode ocr` when elements are not standard UI widgets (e.g. custom-drawn regions).

### Detection Modes

| Mode | Method | Best For |
|------|--------|----------|
| `yolo` (default) | YOLO UI detection → word-split OCR fallback | Menu tabs, button bars, form fields |
| `ocr` | Full-image OCR only | Free-form text, documents, non-standard UI |

#### Why `yolo` mode is the default

Standard OCR engines merge horizontally adjacent text into a single segment. For example, a Windows EHR menu bar like:

```
受付患者一覧  予約患者一覧  枠別予約患者一覧  全枠予約患者一覧  レセプトチェック一覧
```

is returned by full-image OCR as one segment:

```
"受付患者一覧　子約患者一覧　枠別子約患者一覧　全枠子約患者一覧　レセブトチェック一覧"  ← all merged, wrong coordinates
```

In `yolo` mode the pipeline is:

1. **YOLO UI detection** (`foduucom/web-form-ui-field-detection`) — detects individual buttons, inputs, checkboxes. Effective for web-style UIs.
2. **OCR fallback** — if YOLO finds no elements (e.g. Windows desktop apps), the selected OCR backend runs on the full image.

Result with `yolo` mode:

```
"受付患者一覧"  → (464, 462)  ← correct, individual menu item
"予約患者一覧"  → (654, 462)
"枠別予約患者一覧" → ...
```

### OCR Backends

EasyOCR is the default OCR backend used throughout automation. The loaded reader is cached in memory so subsequent calls within the same process have near-zero initialization overhead.

### Command-Line Options

```bash
python -m automation.gui_image_analyzer \
  image.png "search text"                      # Find text (YOLO mode)
  image.png --find-textbox "label"             # Find textbox next to label
  image.png "text" --detection-mode ocr        # Full-image OCR only
  --env-file .env                              # Custom .env path
  --debug                                      # Enable debug logging
```

---

## HDMI Capture Stream Monitor

Real-time video streaming tool for HDMI capture devices with optional UI detection overlay. Runs independently from the chat interface.

### Quick Start

```bash
# Basic streaming (5 FPS, raw mode)
./scripts/run_monitor.sh

# With detection enabled from start
./scripts/run_monitor.sh --detection-on

# Custom frame rate
./scripts/run_monitor.sh --fps 10

# Custom confidence threshold
./scripts/run_monitor.sh --confidence 0.3 --detection-on
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

## Configuration

Edit `.env` file in project root:

```bash
# ESP32 BLE Configuration
ESP32_DEVICE_NAME=BLE Mouse & Keyboard

# Video Capture
CAPTURE_DEVICE_INDEX=0
CAPTURE_WIDTH=1920
CAPTURE_HEIGHT=1080

# Detection / OCR
DETECTION_CONFIDENCE=0.2
DETECTION_IMAGE_SIZE=1024
DETECTION_DEVICE=auto

# OCR
OCR_BACKEND=easyocr       # easyocr (default)
OCR_LANGUAGES=ja,en       # Used by easyocr
OCR_USE_GPU=false         # easyocr only; auto-set to true on Apple Silicon (MPS)

# Detection
DETECTION_MODE=yolo       # yolo (default) or ocr
```

## Troubleshooting

### "ModuleNotFoundError: No module named 'automation'"

Set PYTHONPATH:
```bash
export PYTHONPATH=/path/to/ehr-agentic-toolkit:$PYTHONPATH
```

Or use the helper scripts from project root.

### "No module named 'easyocr'"

Install EasyOCR in the active virtual environment:
```bash
source venv/bin/activate
python -m pip install easyocr
```

### "Failed to connect to mlx_vlm"

Before running `click_history` / `mlx_vlm_history`, ensure the omlx VLM server is running:

```bash
curl -s -H "Authorization: Bearer omlxkey" http://localhost:8000/v1/models
```

### ESP32 Not Connecting

1. Check Bluetooth is enabled on Mac
2. In **Settings > Privacy & Security > Bluetooth**, allow the terminal app you use to run `start_ble_server.sh` / `run_ble_test.sh`
3. Verify ESP32 is powered and advertising
4. Check device name in .env matches ESP32's advertised name
5. Try: `./scripts/run_ble_test.sh`

### Screen Capture Not Working

1. Check capture device is connected and powered
2. Check device index (usually 0, try 1 or 2 if not working)
3. List video devices (macOS): `system_profiler SPCameraDataType`

## Output Files

All outputs are saved to `automation_outputs/`:

- `screenshots/`: Captured screens and debug visualizations
- `logs/`: Detailed logs with timestamps

## Development

### Module Structure

- `config.py`: Configuration and .env loading
- `ble_controller.py`: ESP32 BLE communication
- `ble_server.py`: Long-running BLE server (Unix socket, eliminates per-call connection cost). BLE disconnection is monitored via both callback and periodic health check; when detected, the process exits and returns to the `start_ble_server.sh` restart loop
- `ble_client.py`: Sync client for `ble_server.py`
- `ble_test_cli.py`: Interactive BLE testing CLI tool
- `ehr_input.py`: EHR field input automation (`open_test_patient_chart`, `close_record`, `click_history`, `edit_history`, `input_text_to_field`, `type_kanji_via_ime`, `type_japanese_sentence`, `detect_ime_mode`, `ensure_ime_mode`). Past chart date detection uses the same recognition algorithm as `mlx_vlm_history.py`: reading the date list from the image and estimating coordinates with EasyOCR
- `screen_analyzer.py`: OCR integration (EasyOCR helpers with caching)
- `model_manager.py`: UI detection model management
- `gui_image_analyzer.py`: Image analysis for text coordinates and textbox finding
- `utils.py`: Logging, debugging, progress tracking
- `monitor_stream.py`: HDMI capture stream monitor with YOLO detection
- `local_segmentation.py`: Japanese segmentation using **sudachipy + pykakasi** (main implementation used by `ehr_input.py`)
- `local_segment_probe.py`: Local segmentation CLI probe
- `mlx_vlm_history.py`: Past chart date detection using the omlx VLM server (used by `click_history()`). Reads the date list in the past chart column from top to bottom via the VLM, maps the target date rank to EasyOCR candidate vertical order to determine click coordinates. Template matching for the edit button is also performed when run from CLI
- `mlx_vlm_segmentation.py`: Japanese segmentation helper using the omlx VLM server (reference implementation, also text-only input)
- `mlx_vlm_segment_probe.py`: VLM segmentation CLI probe
- `mlx_vlm_ime.py`: IME conversion candidate reading helper using the omlx VLM server (used by `ehr_input.py`)
- `ehr_composer.py`: End-to-end discharge summary automation — reads past charts, generates summaries via VLM, and inputs them into Word via Notepad IME conversion
- `video_recorder.py`: Phase-specific HDMI video recorder with frame-skip fast-forward for demo video generation

### Adding Features

1. Edit relevant module in `automation/`
2. Test with `--debug` flag
3. Check logs in `automation_outputs/logs/`
