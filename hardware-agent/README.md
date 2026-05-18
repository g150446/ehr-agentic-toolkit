# EHR Agentic Toolkit

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**退院時要約を自動生成・入力する、オンプレミス電子カルテ向け Python 自動化パイプライン。**

HDMI キャプチャで画面を取得し、過去カルテを OCR + VLM でスクロール読み取り、7セクション構成の退院時要約を生成して Word ドキュメントへ自動入力します。すべての処理はローカルで完結し、患者情報は外部に送信されません。

Part of the [EHR Agentic Toolkit](../README.md).

## Key Features

- **Discharge Summary Automation** (`ehr_composer`) — 過去カルテの読み取りから退院時要約の Word 入力まで全自動
- **Privacy-First Design** — 全処理がローカルで完結、患者識別情報は保存・送信なし
- **Universal EHR Compatibility** — HDMI キャプチャ経由でどのオンプレミス EHR にも対応
- **ESP32 BLE Control** — Bluetooth キーボード/マウス HID エミュレーション
- **Zero EHR Modification** — 既存システムへの改変不要
- **Demo Video Recording** (`--movie`) — フェーズ別 MP4 録画（ハッカソンデモ用）

## Project Status

| Component | Status | Description |
|-----------|--------|-------------|
| **ehr_composer (Discharge Summary)** | Complete | 過去カルテ読み取り → サマリ生成 → Word 自動入力 |
| **HDMI Capture** | Complete | MiraBox/対応デバイスからのリアルタイム映像取得 |
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

> **既存クローンに submodule を追加する場合:**
> ```bash
> git submodule update --init
> cd hardware-agent && ./venv/bin/pip install onnxruntime
> ```

## Usage

### Discharge Summary Composer (`ehr_composer`)

退院時要約の生成・入力を全自動で行うメインコマンドです。

**Workflow:**
1. 過去カルテをスクロールしながら OCR + VLM で読み取る
2. 7セクション構成の退院時要約を生成（主訴、現病歴、既往歴、入院後経過、退院時状況、退院時方針、退院時処方）
3. EHR から退院時要約 Word テンプレートを開く
4. ノートパッドで IME 変換 → 切り取り（`Ctrl+X`）→ Word へ貼り付け（`Ctrl+V`）を行ごとに繰り返す

```bash
# 過去カルテを読み取り、退院時要約を生成して Word に入力する（メインコマンド）
python -m automation.ehr_composer --summary

# 前回生成・保存したサマリを再利用して入力のみ実行（カルテ読み取り・生成をスキップ）
python -m automation.ehr_composer --summary-no-scroll

# デモ動画も同時録画
python -m automation.ehr_composer --summary --movie
```

**Summary Persistence**: `--summary` 実行後、生成サマリは `logs/summary_YYYYMMDD_HHMMSS.txt` に自動保存されます。`--summary-no-scroll` は最新の保存済みサマリを読み込むので、入力フェーズだけ再実行できます。

**OCR Backends**:
- 過去カルテ読み取り: **ndlocr-lite** (DEIM + PARSEQ) をデフォルトで使用。`OCR_BACKEND=easyocr` で切り替え可能。
- Word UI ラベル検出（「担当医」等）: **EasyOCR**（固定 — ndlocr-lite は Word の小サイズ印刷フォントを検出できないため）
- IME ポップアップ候補: **ndlocr-lite → EasyOCR → VLM** の cascade

```bash
# 過去カルテ読み取りを EasyOCR に切り替える場合
OCR_BACKEND=easyocr python -m automation.ehr_composer --summary
```

**Demo Video (`--movie`)**: フェーズごとの MP4 を `captures/movie/` に保存します。
- スクロールフェーズ: 3倍速
- UI 操作フェーズ: 等速
- テキスト入力フェーズ: 2倍速
- VLM 処理フェーズ: スキップ（画面変化なし）

---

## Debug / Development Tools

> 以下のツールは `ehr_composer` のデバッグや開発時のトラブルシュートに使用します。通常の運用では不要です。

### omlx VLM Server（前提条件）

`automation.mlx_vlm_history`、`automation.mlx_vlm_segmentation`、`automation.mlx_vlm_ime` はすべて **omlx**（OpenAI 互換 API、ポート 8000）を使用します。事前に omlx サーバーを起動してください。

```bash
# サーバー状態を確認
curl -s -H "Authorization: Bearer omlxkey" http://localhost:8000/v1/models
```

> **Known Issue:** `click_history` / `mlx_vlm_history` では過去カルテ列での日付誤選択が残っています。

---

### EHR Input (`ehr_input`)

テキストや日本語ファイルを BLE キーボード経由で入力するデバッグ用ツールです。`ehr_composer` の入力フェーズの単体テストに使います。

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

HDMI キャプチャデバイスの接続確認や YOLO 検出のリアルタイム可視化に使うデバッグツールです。

```bash
./scripts/run_monitor.sh
./scripts/run_monitor.sh --detection-on
./scripts/run_monitor.sh --fps 10
./scripts/run_monitor.sh --confidence 0.3 --detection-on
```

**Interactive Controls:** Q/ESC — Quit、D — Toggle YOLO、S — Screenshot、F — FPS counter、H — Help、+/- — Confidence

**Output:** Screenshots saved to `monitor_outputs/`, logs in `automation_outputs/logs/`

---

### HDMI Snapshot Capture

HDMI キャプチャデバイスから静止画を1枚保存します。

```bash
python scripts/capture_windows.py
python scripts/capture_windows.py myshot.jpg
```

**Output:** All images are saved to the `captures/` directory.

---

### Past Chart Column OCR / Layout Comparison

保存済み画像に対して OCR 戦略の比較分析を行うデバッグツールです。

```bash
./scripts/run_history_panel_analyzer.sh captures/0410.jpg --date 20260410
```

Output is saved to `automation_outputs/history_panel_analysis/<run-name>/`.

---

### BLE Test CLI

ESP32 BLE キーボード/マウスの単体テストツールです。

```bash
python -m automation.ble_test_cli
```

---

### GUI Image Analyzer

スクリーンショットからテキスト座標やテキストボックス位置を検出するデバッグツールです。

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
