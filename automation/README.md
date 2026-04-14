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
> `click_history` / `mlx_vlm_history` を使う前には `bash scripts/start_mlx_vlm_server.sh qwen` も別ターミナルで起動する。

### コマンドライン使い方

`automation.ehr_input` はコマンドライン引数によって動作を切り替えます。

```bash
# 引数なし: テスト患者カルテを開く
python -m automation.ehr_input

# 日本語テキスト: IME変換のみ実行（カルテは開かない）
python -m automation.ehr_input 肺炎

# 英語テキスト: 英数字モードで直接入力
python -m automation.ehr_input tesuto

# テキストファイル: 内容を読み込んで入力
python -m automation.ehr_input data/patient_records/asthma_1.txt

# 日英混在テキスト: 文節ごとに IME モードを自動切替
python -m automation.ehr_input "COVID-19の感染を確認した"

# 取り消し[F9]ボタンをクリックしてカルテを閉じる
python -m automation.ehr_input "close record"

# 第一引数が "open test"、第二引数がテキスト: カルテを開いてから入力
python -m automation.ehr_input "open test" 肺炎
python -m automation.ehr_input "open test" "MRI所見"
python -m automation.ehr_input "open test" data/patient_records/asthma_1.txt
```

日本語テキストを渡すと、`ehr_input.py` は **Qwen 3.5 4B MLX を優先して文節分割**し、ローマ字はローカル辞書で補正してから IME 入力します。Qwen の分割が細かすぎて IME 候補を不安定化させる場合は、`sudachipy + pykakasi` のローカル分割へ自動フォールバックします。引数が読み取り可能なテキストファイルなら、その**ファイル内容**を同じ入力フローに流します。

4文字を超える文章や助詞を含む文は `type_japanese_sentence()` で文節単位に分割して**逐次入力**します。句読点（`、` → `,` / `。` → `.` + Enter）に加えて、改行・`[` `]` `(` `)` `%` `:` も専用キー送信に切り替えて処理します。

日英混在テキスト（例: `"COVID-19の感染を確認した"`）では、ASCII のみの文節は英数字モード、日本語文節はひらがなモードで入力するよう IME を自動切替します。

> **既知の問題**: `data/patient_records/asthma_1.txt` の再検証では、空欄のまま止まる問題は解消しましたが、`咽頭痛` / `昨晩` / `咳嗽` 付近の誤変換がまだ残ります。実画面では本文先頭まで入力が進むことを確認済みですが、完全自動入力としては未解決です。

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

既定では `mlx-community/Qwen3.5-4B-MLX-4bit` を使う LLM ベースの実装です。`ehr_input.py` では使用していませんが、LLM の出力比較などに利用できます。事前に `bash scripts/start_mlx_vlm_server.sh qwen` でサーバーを起動してください。名前に `mlx_vlm` を含みますが、この用途では画像は送らずテキストだけを `/v1/chat/completions` に渡します。

```bash
python -m automation.mlx_vlm_segment_probe "肺炎に対して抗菌薬による治療を行う"
```

### Ollama 文節分割プローブ（参考実装）

Ollama (`gemma4:e2b`) を使った実装です。`ehr_input.py` では使用していませんが、Ollama の動作確認に利用できます。

```bash
python -m automation.ollama_segment_probe "肺炎に対して抗菌薬による治療を行う"
```

### click_history

過去カルテ列から指定日付のエントリを検出してクリックする。

1. HDMIスクリーンをキャプチャしてOCRを実行
2. EasyOCR で日付候補の座標を抽出し、`mlx_vlm.server` 上の Qwen 3.5 4B に **過去カルテ欄画像 + 対象日付** を渡して、見えている日付一覧を上から順に読ませる
   - 正規表現で一意に特定できる場合はVLM不要（高速パス）
   - Qwen が返した **対象日付の順位** を、EasyOCR 候補の縦順へ対応づけてクリック座標を決める
   - `mlx_vlm_history.py` と同じ、**画像で日付一覧を読取り / EasyOCRで座標推定** の認識アルゴリズムを使う
3. 検出した座標にBLEマウスを移動してクリック

> **前提**: `bash scripts/start_mlx_vlm_server.sh qwen` を起動しておく。

> **既知の問題**: `click_history` / `mlx_vlm_history` は、過去カルテ欄の日付誤選択がまだ解消していません。現時点では未解決です。

```bash
python -m automation.ehr_input "click history 20260312"
```

```python
from automation.ehr_input import click_history
click_history("20260312")
```

環境変数でVLMの接続先・モデル・タイムアウトを変更できる:

| 環境変数 | デフォルト | 説明 |
|---------|-----------|------|
| `MLX_VLM_HISTORY_URL` | `http://localhost:8181/v1/chat/completions` | multimodal chat completion endpoint (`mlx_vlm.server`) |
| `MLX_VLM_HISTORY_MODEL` | `mlx-community/Qwen3.5-4B-MLX-4bit` | 使用モデル |
| `MLX_VLM_HISTORY_TIMEOUT` | `120` | タイムアウト秒数 |

### edit_history

過去カルテ列の指定日付エントリをクリックし、1秒待機後に修正ボタンをクリックして修正モードへ移行する。

1. `click_history(date_str)` を呼び出して対象日付のエントリをクリック
2. 1秒待機（修正ボタンが表示されるまで）
3. HDMIスクリーンを再キャプチャ
4. OpenCV テンプレートマッチング（`match_templates/edit_button.jpg`）で修正ボタンを検出
5. 検出した座標にBLEマウスを移動してクリック

```bash
python -m automation.ehr_input "edit history 20260312"
```

```python
from automation.ehr_input import edit_history
edit_history("20260312")
```

> **テンプレート画像**: `match_templates/edit_button.jpg` に修正ボタンの切り取り画像が必要。
> マッチングスコアが 0.7 未満の場合は `RuntimeError` を送出する。

#### テンプレートマッチングの動作確認（クリックなし）

`mlx_vlm_history.py` の CLI は、保存画像に対して **full-image EasyOCR** で候補位置を作り、`mlx_vlm.server` には **過去カルテ欄の日付一覧を上から順に読ませる**。その順位を EasyOCR 座標へ対応づけて日付座標を特定する。`click_history()` も同じ認識アルゴリズムを使う。その後、修正ボタンのテンプレートマッチングも実行して座標を表示する（クリックはしない）。

```bash
python -m automation.mlx_vlm_history captures/history.jpg 20260312
# → 日付座標 + 修正ボタン座標とマッチングスコアを表示
```

### 過去カルテ列の解析

保存画像に対して、**OCRアンカーで過去カルテ列 ROI を推定**しつつ、EasyOCR full-image / EasyOCR + UI detection を比較する。

```bash
# EasyOCR full-image / UI detection 比較
python -m automation.history_panel_analyzer captures/0410.jpg --date 20260410

# helper script から実行
./scripts/run_history_panel_analyzer.sh captures/0410.jpg --date 20260410
```

デフォルトで次を比較する:

1. `EasyOCR + full-image OCR`
2. `EasyOCR + UI detection OCR`

出力先は `automation_outputs/history_panel_analysis/<run-name>/`。主な生成物:

- `*_annotated.png`: 候補日付ボックスと推定 ROI
- `*_summary.txt`: 各戦略の OCR セグメント数・日付候補数・一致候補
- `history_roi.png`: 推定した過去カルテ列 ROI
- `summary.txt`: 全体比較サマリと推奨
- `manifest.json`: 機械読取しやすい比較結果

### close_record

画面右上の「取消[F9]」ボタンをOCRで検出してクリックし、開いているカルテを閉じる。

1. HDMIスクリーンをキャプチャしてOCRで「取消」テキストを検索
2. 検出した座標にBLEマウスを移動してクリック

```python
from automation.ehr_input import close_record
close_record()
```

### open_test_patient_chart

テスト患者のカルテを自動で開く。以下の手順を実行:

0. HDMIスクリーンをキャプチャしてOCRで「患者検索」タブを検出しクリック（0.5秒待機）。タブが既に選択済みの場合は青文字になりOCRで検出できないため、検出できなかった場合はスキップして次のステップへ進む
1. フリガナ欄をOCRで検索してクリックし `tesuto` を入力、Enter → 患者一覧を表示
2. 0.5秒待って Enter → 先頭患者を選択してカルテを開く
3. 2秒待って Enter → 表示直後のダイアログを閉じる

```python
from automation.ehr_input import open_test_patient_chart
open_test_patient_chart()
```

### type_kanji_via_ime

ローマ字をIMEで変換し、HDMIキャプチャから切り出した候補画像を **Qwen 3.5 4B MLX 優先 / OCR フォールバック** で確認してから Enter で確定する。一致未確認の候補をそのまま Enter で確定することはしない。

**IME候補の検出方法**: 画面全体のOCRではなく、Windowsが変換候補を**黒背景・白文字**で反転表示する特徴をOpenCVで検出し、その領域だけをOCRすることで元々画面に存在する同じ漢字との誤検知を防ぐ。

**Qwen プロンプト**: 目標文字列を先頭に明示し「黒く反転（ハイライト）されている行の文字列だけを読み取ってください」と指示することで、候補ウィンドウ内の誤読を低減している。

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

**`detect_ime_mode(frame)`**: 画面下部（タスクバー付近・下100px）に対して OpenCV テンプレートマッチング（`TM_CCOEFF_NORMED`）を行い、`match_templates/english_ime.png` と `match_templates/hiragana_ime.png` のスコアを比較する。閾値 0.7 を超えた方のモードを返し、どちらも届かなければ `None` を返す。EasyOCR は使用しない。

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

### "mlx_vlmへの接続に失敗しました"

`click_history` / `mlx_vlm_history` の前に、別ターミナルで MLX VLM サーバーを起動してください:

```bash
bash scripts/start_mlx_vlm_server.sh qwen
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
- `ble_server.py`: Long-running BLE server (Unix socket, eliminates per-call connection cost). BLE切断は callback と定期ヘルスチェックの両方で監視し、検知時はプロセスを終了して `start_ble_server.sh` の再起動ループへ返す
- `ble_client.py`: Sync client for `ble_server.py`
- `ble_test_cli.py`: Interactive BLE testing CLI tool
- `ehr_input.py`: EHR field input automation (`open_test_patient_chart`, `close_record`, `click_history`, `edit_history`, `input_text_to_field`, `type_kanji_via_ime`, `type_japanese_sentence`, `detect_ime_mode`, `ensure_ime_mode`)。過去カルテ日付検出は `mlx_vlm_history.py` と同じ、画像で日付一覧を読取りして EasyOCR で座標推定する認識アルゴリズムを使う
- `screen_analyzer.py`: OCR integration (EasyOCR helpers with caching)
- `model_manager.py`: UI detection model management
- `gui_image_analyzer.py`: Image analysis for text coordinates and textbox finding
- `utils.py`: Logging, debugging, progress tracking
- `monitor_stream.py`: HDMI capture stream monitor with YOLO detection
- `local_segmentation.py`: **sudachipy + pykakasi** による日本語文節分割（`ehr_input.py` が使用するメイン実装）
- `local_segment_probe.py`: ローカル文節分割 CLI プローブ
- `mlx_vlm_history.py`: `mlx_vlm.server` を使う過去カルテ日付検出（`click_history()` が使用）。Qwen に過去カルテ欄の日付一覧を上から順に読ませ、対象日付の順位を EasyOCR 候補の縦順へ対応づけてクリック座標を決める。CLI実行時は修正ボタンのテンプレートマッチングも実施
- `mlx_vlm_segmentation.py`: `mlx_vlm.server` を使った日本語文節分割ヘルパー (参考実装、こちらもテキスト入力のみ)
- `mlx_vlm_segment_probe.py`: mlx_vlm 文節分割 CLI プローブ
- `ollama_segmentation.py`: Ollama を使った日本語文節分割ヘルパー (参考実装)
- `ollama_segment_probe.py`: Ollama 文節分割 CLI プローブ

### Adding Features

1. Edit relevant module in `automation/`
2. Test with `--debug` flag
3. Check logs in `automation_outputs/logs/`
