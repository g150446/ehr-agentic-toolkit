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

### コマンドライン使い方

`automation.ehr_input` はコマンドライン引数によって動作を切り替えます。

```bash
# 引数なし: テスト患者カルテを開く
python -m automation.ehr_input

# 日本語テキスト: IME変換のみ実行（カルテは開かない）
python -m automation.ehr_input 肺炎

# 英語テキスト: 英数字モードで直接入力
python -m automation.ehr_input tesuto

# 日英混在テキスト: 文節ごとに IME モードを自動切替
python -m automation.ehr_input "COVID-19の感染を確認した"

# 第一引数が "open test"、第二引数がテキスト: カルテを開いてから入力
python -m automation.ehr_input "open test" 肺炎
python -m automation.ehr_input "open test" "MRI所見"
```

日本語テキストを渡すと、**sudachipy + pykakasi** で文節分割・ローマ字変換してから IME 入力します。辞書ベースの決定論的変換なので長母音（例: `治療` → `chiryou`）も正確です。

4文字を超える文章や助詞を含む文は `type_japanese_sentence()` で文節単位に分割して入力します。句読点（`、` → `,` / `。` → `.` + Enter）も自動処理します。

日英混在テキスト（例: `"COVID-19の感染を確認した"`）では、ASCII のみの文節は英数字モード、日本語文節はひらがなモードで入力するよう IME を自動切替します。

### ローカル文節分割プローブ（推奨）

**sudachipy + pykakasi** を使ったローカル文節分割の動作確認ツールです。外部サービス不要・長母音も正確です。

```bash
python -m automation.local_segment_probe "肺炎に対して、抗菌薬による治療を行う。"
```

出力例:

```
対象文: '肺炎に対して、抗菌薬による治療を行う。'
エンジン: sudachipy (SplitMode.C) + pykakasi (hepburn)
分割サマリ: 肺炎(haien) / に(ni) / 対して(taishite) / 、(,) / 抗菌薬(koukinyaku) / に(ni) / よる(yoru) / 治療(chiryou) / を(wo) / 行う(okonau) / 。(.)
分割結果:
  1. '肺炎' (haien)
  2. 'に' (ni)
  3. '対して' (taishite)
  4. '、' (,)
  5. '抗菌薬' (koukinyaku)
  ...
```

### mlx_vlm 文節分割プローブ（参考実装）

`mlx-community/gemma-4-e2b-it-4bit` を使った LLM ベースの実装です。`ehr_input.py` では使用していませんが、LLM の出力比較などに利用できます。事前に `bash scripts/start_mlx_vlm_server.sh` でサーバーを起動してください。

```bash
python -m automation.mlx_vlm_segment_probe "肺炎に対して抗菌薬による治療を行う"
```

### Ollama 文節分割プローブ（参考実装）

Ollama (`gemma4:e2b`) を使った実装です。`ehr_input.py` では使用していませんが、Ollama の動作確認に利用できます。

```bash
python -m automation.ollama_segment_probe "肺炎に対して抗菌薬による治療を行う"
```

### open_test_patient_chart

テスト患者のカルテを自動で開く。以下の手順を実行:

1. HDMIスクリーンをキャプチャしてフリガナ欄をOCRで検索
2. 欄をクリックして `tesuto` を入力し Enter → 患者一覧を表示
3. 0.5秒待って Enter → 先頭患者を選択してカルテを開く
4. 1秒待って Enter → 表示直後のダイアログを閉じる

```python
from automation.ehr_input import open_test_patient_chart
open_test_patient_chart()
```

### type_kanji_via_ime

ローマ字をIMEで変換し、HDMIキャプチャ＋OCRで候補を確認してから Enter で確定する。

**IME候補の検出方法**: 画面全体のOCRではなく、Windowsが変換候補を**黒背景・白文字**で反転表示する特徴をOpenCVで検出し、その領域だけをOCRすることで元々画面に存在する同じ漢字との誤検知を防ぐ。

```python
from automation.ehr_input import type_kanji_via_ime

# "haien" と入力し、IME候補で "肺炎" を確認してEnterで確定
type_kanji_via_ime("haien", "肺炎")

# ローマ字を自動変換して実行することも可能
from automation.ehr_input import _kanji_to_romaji
romaji = _kanji_to_romaji("肺炎")  # → "haien"
type_kanji_via_ime(romaji, "肺炎")
```

### detect_ime_mode / ensure_ime_mode

Windows IME の現在入力モードをスクリーンキャプチャから判定し、必要に応じて切替える。

**`detect_ime_mode(frame)`**: 画面下部（タスクバー）を OCR し、「あ」が検出されれば `'japanese'`、「A」/「Ａ」が検出されれば `'english'`、判定不能なら `None` を返す。

**`ensure_ime_mode(target_mode, client, current_mode)`**: 現在モードが目標と異なる場合のみ `key:zenkaku`（半角/全角キー）を送信してトグルし、新しいモード文字列を返す。画面再キャプチャはしない設計で、呼び出し元がモードをトラッキングする。

```python
from automation.ehr_input import detect_ime_mode, ensure_ime_mode
from automation.screen_analyzer import capture_screen
from automation.ble_client import BLEClient

frame = capture_screen(0)
current = detect_ime_mode(frame)       # 'japanese' / 'english' / None
client = BLEClient()
current = ensure_ime_mode("english", client, current)  # 必要なら半角/全角を送信
```

### input_text_to_field

ラベル付き入力欄をOCRで検索してテキストを入力する低レベル関数。

```python
from automation.ehr_input import input_text_to_field

# フリガナ欄に "tesuto" を入力
input_text_to_field(input_text="tesuto", label="フリガナ")
```

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
- **Fast OCR** - RapidOCR (ONNX Runtime) as default backend, 3–10× faster than EasyOCR on CPU/M1
- **YOLO UI Detection** - Detects individual UI elements (buttons, tabs, inputs) before OCR to prevent menu items from being merged into one text segment

### How It Works

**YOLO mode (default):**
1. **UI Element Detection** - `foduucom/web-form-ui-field-detection` (YOLOv8) detects individual UI elements
2. **Per-element OCR** - Each detected element is cropped and OCR'd separately
3. **Label Matching** - Finds the label text among individually recognized elements
4. **Textbox Search** - Looks for element to the right within vertical tolerance
5. **Coordinate Output** - Returns center (x, y) coordinates

**OCR mode (`--detection-mode ocr`):**
1. RapidOCR / EasyOCR runs on the full image at once
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
2. **Word-split OCR fallback** — if YOLO finds no elements (e.g. Windows desktop apps), RapidOCR runs with character-level bounding boxes (`return_word_box=True`). Characters are then grouped into separate segments wherever the horizontal gap between them exceeds 1.5× the average character width of that line.

Result with `yolo` mode:

```
"受付患者一覧"  → (464, 462)  ← correct, individual menu item
"予約患者一覧"  → (654, 462)
"枠別予約患者一覧" → ...
```

### OCR Backends

Two OCR backends are supported, selectable via `OCR_BACKEND` in `.env`:

| Backend | Library | Speed | Japanese Model |
|---------|---------|-------|----------------|
| `rapidocr` (default) | RapidOCR + ONNX Runtime | Fast (CPU/M1 optimized) | PP-OCRv4 Japan model |
| `easyocr` | EasyOCR + PyTorch | Slower | Built-in ja+en model |

RapidOCR downloads the Japanese model (~9 MB) on first run and caches it. Both backends cache the loaded reader in memory so subsequent calls within the same process have near-zero initialization overhead.

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

Real-time video streaming tool for HDMI capture devices with optional DocLayout-YOLO detection overlay. Runs independently from the chat interface.

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

# DocLayout-YOLO
DOCLAYOUT_MODEL_PATH=./DocLayout-YOLO/models/doclayout_docstructbench.pt
DETECTION_CONFIDENCE=0.2
DETECTION_IMAGE_SIZE=1024
DETECTION_DEVICE=auto

# OCR
OCR_BACKEND=rapidocr      # rapidocr (default, faster) or easyocr
OCR_LANGUAGES=ja,en       # Used by easyocr backend only
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
model.save('DocLayout-YOLO/models/doclayout_docstructbench.pt')
"
```

### ESP32 Not Connecting

1. Check Bluetooth is enabled on Mac
2. Verify ESP32 is powered and advertising
3. Check device name in .env matches ESP32's advertised name
4. Try: `./scripts/run_ble_test.sh`

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
- `ble_server.py`: Long-running BLE server (Unix socket, eliminates per-call connection cost)
- `ble_client.py`: Sync client for `ble_server.py`
- `ble_test_cli.py`: Interactive BLE testing CLI tool
- `ehr_input.py`: EHR field input automation (`open_test_patient_chart`, `input_text_to_field`, `type_kanji_via_ime`, `type_japanese_sentence`, `detect_ime_mode`, `ensure_ime_mode`)
- `screen_analyzer.py`: DocLayout-YOLO + OCR integration (RapidOCR/EasyOCR with caching)
- `model_manager.py`: Multi-model management (DocLayout-YOLO + YOLOv11)
- `gui_image_analyzer.py`: Image analysis for text coordinates and textbox finding
- `utils.py`: Logging, debugging, progress tracking
- `monitor_stream.py`: HDMI capture stream monitor with YOLO detection
- `local_segmentation.py`: **sudachipy + pykakasi** による日本語文節分割（`ehr_input.py` が使用するメイン実装）
- `local_segment_probe.py`: ローカル文節分割 CLI プローブ
- `mlx_vlm_segmentation.py`: mlx_vlm サーバーを使った日本語文節分割ヘルパー (参考実装)
- `mlx_vlm_segment_probe.py`: mlx_vlm 文節分割 CLI プローブ
- `ollama_segmentation.py`: Ollama を使った日本語文節分割ヘルパー (参考実装)
- `ollama_segment_probe.py`: Ollama 文節分割 CLI プローブ

### Adding Features

1. Edit relevant module in `automation/`
2. Test with `--debug` flag
3. Check logs in `automation_outputs/logs/`
