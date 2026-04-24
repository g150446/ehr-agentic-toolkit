# EHR Agentic Toolkit

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**AI-powered clinical decision support bridge for on-premises Electronic Health Record systems.**

EHR Agentic Toolkit connects your existing on-premises EHR system with AI capabilities without requiring direct system integration. Using screen capture and OCR technology, it extracts clinical information, anonymizes patient data, and provides AI-assisted clinical summaries, differential diagnoses, and treatment suggestions.

## ✨ Key Features

- 🏥 **Universal EHR Compatibility** - Works with any on-premises EHR via HDMI capture
- 🔒 **Privacy-First Design** - All processing happens locally, patient identifiers never stored
- 🤖 **AI-Assisted Clinical Support** - Summaries, differential diagnosis, treatment suggestions (planned)
- 🔌 **Plugin Architecture** - Easy to add support for new EHR systems
- 🛡️ **Enterprise-Grade Security** - Encrypted storage, audit logging, HIPAA-conscious design
- 🎯 **Zero EHR Modification** - No changes to existing systems required
- 📹 **Real-time Stream Monitor** - Debug HDMI capture with YOLO detection visualization ✅ **Implemented**
- 🎮 **ESP32 BLE Control** - Keyboard/mouse HID emulation over Bluetooth ✅ **Implemented**
- 🖼️ **GUI Image Analyzer** - Find text coordinates and textbox positions in screenshots ✅ **Implemented**
- 🧪 **BLE Test CLI** - Interactive testing tool for ESP32 keyboard/mouse control ✅ **Implemented**

## 📊 Project Status

| Component | Status | Description |
|-----------|--------|-------------|
| **HDMI Capture** | ✅ **Complete** | Real-time video capture from MiraBox/compatible devices |
| **Layout Analysis** | 🔄 **In Progress** | Evaluating ROI inference and detector-first OCR for EHR layout parsing |
| **OCR (EasyOCR)** | ✅ **Complete** | Multi-language text extraction with EasyOCR as the default path |
| **Stream Monitor** | ✅ **Complete** | Interactive HDMI capture monitor with detection overlay |
| **ESP32 BLE Control** | ✅ **Complete** | Keyboard/mouse HID emulation over Bluetooth |
| **BLE Test CLI** | ✅ **Complete** | Interactive testing tool for ESP32 keyboard/mouse |
| **GUI Image Analyzer** | ✅ **Complete** | Text coordinate detection and textbox finding |
| **EHR Adapters** | 🔄 **In Progress** | Fujitsu adapter framework implemented |
| **Anonymization** | 📋 **Planned** | PHI removal and data anonymization |
| **Encrypted Storage** | 📋 **Planned** | AES-256 encrypted PostgreSQL database |
| **AI Engine** | 📋 **Planned** | Claude API / local LLM integration |
| **Clinical Decision Support** | 📋 **Planned** | Differential diagnosis, treatment suggestions |

**Current Focus:** Building automation infrastructure and screen capture pipeline.
**Next Steps:** EHR adapter development and anonymization layer.

## 🏗️ Architecture
```
┌─────────────────┐
│  EHR System     │
│  (On-Premises)  │
└────────┬────────┘
         │ HDMI
         ▼
┌─────────────────┐
│ Capture Layer   │◄── Screen Capture & OCR
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Adapter Layer   │◄── EHR-specific Parsing
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Anonymization   │◄── Remove PHI
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Encrypted DB    │◄── Secure Storage
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ AI Engine       │◄── Claude/Local LLM
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Clinical Output │◄── Decision Support
└─────────────────┘
```

## 🚀 Quick Start

### Prerequisites

**Hardware:**
- macOS 11+ (M1 or later recommended) or Linux
- HDMI capture device (e.g., MiraBox, Elgato)
- ESP32 module (for Windows automation, optional)
- External SSD (for encrypted storage, optional)

**Software:**
- Python 3.10+
- PostgreSQL 15+ (for production use, optional)

### Installation

**Option 1: Automated Setup (Recommended)**
```bash
# Clone the repository
git clone https://github.com/g150446/ehr-agentic-toolkit.git
cd ehr-agentic-toolkit

# Run setup script (installs everything)
./scripts/setup_automation.sh

# Edit configuration
cp .env.example .env
nano .env  # Configure your settings
```

**Option 2: Manual Setup**
```bash
# Clone and enter directory
git clone https://github.com/g150446/ehr-agentic-toolkit.git
cd ehr-agentic-toolkit

# Create virtual environment (Python 3.12 recommended)
python3.12 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create .env file
cp .env.example .env
```

## 💻 Usage

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

### HDMI スナップショットキャプチャ

HDMIキャプチャデバイスから1枚の静止画を保存するシンプルなキャプチャツール。

```bash
# タイムスタンプ付きファイル名で保存（captures/windows_capture_YYYYMMDD_HHMMSS.jpg）
python scripts/capture_windows.py

# ファイル名を指定して保存（captures/ に保存）
python scripts/capture_windows.py myshot.jpg

# 拡張子を省略しても .jpg が自動付与される
python scripts/capture_windows.py myshot
```

**出力先:** すべての画像は `captures/` ディレクトリに保存されます。

---

### 過去カルテ列の OCR / レイアウト比較

保存画像に対して、過去カルテ列向けの **OCR アンカー ROI 推定** と **OCR / レイアウト戦略比較** を実行できます。

```bash
./scripts/run_history_panel_analyzer.sh captures/0410.jpg --date 20260410
```

このコマンドは次を比較します。

- EasyOCR + 全画面 OCR
- EasyOCR + UI detection OCR

出力は `automation_outputs/history_panel_analysis/<run-name>/` に保存されます。

---

### omlx VLM サーバー

`automation.mlx_vlm_history`、`automation.mlx_vlm_segmentation`、`automation.mlx_vlm_ime` はすべて **omlx** (OpenAI 互換 API、ポート 8000) を使います。事前に omlx サーバーを起動しておいてください。

```bash
# omlx サーバーの状態確認
curl -s -H "Authorization: Bearer omlxkey" http://localhost:8000/v1/models
```

> **既知の問題**: `click_history` / `mlx_vlm_history` は、過去カルテ欄の日付誤選択がまだ残っており、現時点では未解決です。

### EHR Input でテキストファイルを入力

`automation.ehr_input` は通常の文章だけでなく、**テキストファイルのパス**も受け取れます。読み取り可能なファイルが指定された場合、その内容を既存の日本語/英語/混在入力フローで遠隔キーボード送信します。

現在の `ehr_input` は、長文入力時に **Qwen 3.5 4B MLX を優先して日本語セグメントを切り出し**、ローマ字化はローカル辞書で補正しながら逐次入力します。IME 候補確認も Qwen 優先で行い、一致未確認の候補を盲目的に Enter で確定しないようにしています。**カタカナ部分は mixed セグメント中でも切り出して、常に F7 による全角カタカナ変換で確定**するため、`アレルギー性` のような語でもカタカナ部分を漢字変換候補に流しません。BLE で扱いにくい記号は入力前に可読な代替へ正規化しており、たとえば `℃` は `度` として送信します。さらに helper reset では、**コマンド開始時に patient_record の第3ペイン座標を 1 回だけ検出して VLM 検証し、以後の Escape compare では同じ crop 座標を再利用**します。比較基準の文字列側も、**最後の日本語 anchor に直後の確定済み ASCII/記号 suffix を連結した anchor tail**（例: `症状(`）を保持するため、Escape 後にその suffix が残っていても正常状態として扱います。

日本語全角でそのまま表示したい記号は、現在 **`、` `。` `・` `ー` `〜` `「` `」` `『` `』`** を特別扱いしています。`、` `。` `・` `〜` `「` `」` `『` `』` は前後の単語から**単独トークンとして分離**し、日本語モードで対応キーを送った直後に **Enter で明示確定**します。**長音記号 `ー` は単独で現れる場合だけ同様に即時確定し、ひらがな・カタカナの直後に続く場合はその語の一部として保持**して一緒に変換・確定します（例: `コーテフ`, `えーと`, `アレルギー`）。その他の全角記号（例: `（` `）` `％` `：` `［` `］` `【` `】`）は、今のところ **半角 ASCII に正規化して送信**します。

```bash
python -m automation.ehr_input data/patient_records/asthma_1.txt
python -m automation.ehr_input "open test" data/patient_records/asthma_1.txt
python -m automation.ehr_input --win10 data/patient_records/asthma_1.txt   # Windows 10 環境
```

ヘルプを表示するには:

```bash
python -m automation.ehr_input --help
```

#### IME モード検出

入力時に `a` を 1 文字タイプし、スクリーンキャプチャを Qwen3-VL に渡して `a`（英語モード）か `あ`（日本語モード）かを判定します。cleanup は英語モードでは Backspace、日本語モードでは Escape 後に **未確定組成が残っている場合だけ** Backspace を送り、直前に確定済みの文字を壊さないようにしています。Windows バージョン固有の動作には `--win10` フラグを使用します。

| オプション | 説明 |
|---|---|
| `--win10` | Windows 10 固有の動作（インライン変換スキップなど）を有効化 |
| `--clear` | 入力前に Backspace を 50 回送信してフィールドをクリア |
| `--fireworks <model>` | 文節分割・IME 候補読取・ヘルパー単語提案を Fireworks AI の指定モデルへ切り替える |
| `--google-ai-studio` | 文節分割・IME 候補読取・ヘルパー単語提案を Google AI Studio の `gemma-4-26b-a4b-it` へ切り替える |
| `--novita [model]` | 文節分割・IME 候補読取・ヘルパー単語提案を Novita AI へ切り替える。モデル省略時は `google/gemma-4-31b-it` を使う |
| `--openrouter <model>` | 文節分割・IME 候補読取・ヘルパー単語提案を OpenRouter の vision 対応モデルへ切り替える |
| `--mactest` | HDMI/BLE の代わりに Mac ローカル画面 + `pyautogui` で動作確認する |

#### 保守運用メモ

- 各実行のログは `logs/*.txt` に保存され、先頭に **実行ファイル名・生のコマンドライン・解析済みオプション要約** が記録されます。
- 変換が期待どおりでない場合は、`[VLM一致]`、`[候補照合/romaji]`、`[試行N]`、`[ヘルパー単語]` の行を順に追うと、**候補番号の誤選択なのか / 候補未発見なのか / 後続フォールバックへ落ちたのか** を切り分けやすくなります。
- pure kanji ターゲットでは、**読みだけ一致する mixed 候補を即採用しない**実装になっており、曖昧な候補はヘルパー単語 fallback やハイライト候補確認へ回します。
- ヘルパー単語 fallback に入る直前は、各 `Escape` 後に画面を再キャプチャし、**直前の正常確定時に保存した baseline 画像**と **Esc 後の current 画像**を VLM に比較させます。patient_record では、コマンド開始時に **第3ペイン座標を一度だけ検出して VLM で妥当性確認**し、その後の compare crop は毎回その固定座標を再利用します。**Windows Notepad** と判定された場合は従来どおり Notepad 本文領域だけを切り出して上メニュー帯と Windows タスクバーを除外します。比較時は **最後の確定済み文字列** を基準に見ており、日本語 anchor の直後に確定済み ASCII/記号 suffix がある場合はそれも anchor tail に含めます（例: `症状(`）。このため、`症状(` のような正常末尾を保ったまま未確定部分だけが消えたケースを正しく reset 完了として扱えます。`captures/` には `debug_panel_detection_*`、`debug_helper_reset_*_compare_crop.png`、`debug_vlm_input_helper_reset_compare_*` が残り、`logs/*.txt` の `[helper reset][compare]` 行で各 `Esc` 後の yes/no 判定を確認できます。

> **既知の問題**: `data/patient_records/asthma_1.txt` での再検証では、空白のまま止まる問題は解消しましたが、`咽頭痛` のような語でまだ誤変換が残ります。特に長文冒頭では `[` のような記号未反映や、`昨晩` / `咳嗽` 付近の変換揺れが残っています。

---

### BLE Test CLI

ESP32 BLE キーボード/マウスの単体確認には `automation.ble_test_cli` を使います。

```bash
python -m automation.ble_test_cli
```

接続後、Escape キーの確認は次のどちらでも実行できます。

```text
press esc
esc
```

`press escape` も `esc` に正規化されて同じ動作になります。

---

### GUI Image Analyzer

Analyze screenshots to find text coordinates and textbox positions for GUI automation.

**Find Text Coordinates:**
```bash
# Find coordinates of specific text
python -m automation.gui_image_analyzer screenshot.png "患者検索"
# Output: 📍 Text "患者検索" found at coordinates: (x=80, y=462)
```

**Find Textbox Next to Label:**
```bash
# Find textbox to the right of a label
python -m automation.gui_image_analyzer screenshot.png --find-textbox "フリガナ"
# Output: 📍 Textbox right of "フリガナ" detected visually at: (x=333, y=684)

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
# Initialize database
./scripts/start_postgresql.sh
ehr-bridge init

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

## 🏥 Supported EHR Systems

| EHR System | Status | Adapter |
|------------|--------|---------|
| Fujitsu EHR | ✅ Supported | Built-in |
| NEC MegaOak HR | 🔄 In Progress | Community |
| Philips Tasy | 🔄 In Progress | Community |
| Medicom | 📋 Planned | - |
| Custom/Generic | ✅ Supported | Configuration-based |

Don't see your EHR? Create a custom adapter using our [adapter development guide](docs/custom-ehr-setup.md).

## 📖 Documentation

**User Guides:**
- [Getting Started Guide](docs/getting-started.md)
- [Automation Tools Guide](automation/README.md) ⭐ **Start here for automation**
- [Script Reference](scripts/README.md)

**Technical Documentation:**
- [Architecture Overview](docs/architecture.md)
- [Security Guidelines](docs/security-guidelines.md)
- [EHR Configuration Guide](docs/ehr-configuration-guide.md)
- [API Reference](docs/api-reference.md)
- [Custom Adapter Development](docs/custom-ehr-setup.md)

**Development:**
- [CLAUDE.md](CLAUDE.md) - Instructions for Claude Code

## 🔧 Troubleshooting

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

#### ESP32 Not Connecting

**Checklist:**
1. ✅ Bluetooth enabled on Mac/PC
2. ✅ ESP32 powered and advertising
3. ✅ Device name in `.env` matches ESP32's advertised name
4. ✅ Test connection: `./scripts/run_ble_test.sh`

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

## 🔒 Security & Privacy

This toolkit is designed with healthcare privacy regulations in mind:

- ✅ **No PHI Storage**: Patient identifiable information is never persisted
- ✅ **Local Processing**: All AI processing can run entirely offline
- ✅ **Encrypted Storage**: AES-256 encryption for all stored data
- ✅ **Audit Logging**: Comprehensive activity logs for compliance
- ✅ **Anonymization**: Age ranges and gender only, no names or IDs

See [Security Guidelines](docs/security-guidelines.md) for detailed information.

## 🛠️ Development
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

## 🤝 Contributing

Contributions are welcome! Please read our [Contributing Guide](CONTRIBUTING.md) first.

### Adding Support for a New EHR

1. Create adapter in `ehr_ai_bridge/adapters/your_ehr/`
2. Implement `BaseEHRAdapter` interface
3. Add configuration YAML
4. Write tests
5. Submit PR with documentation

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This software is provided for research and development purposes. It is not a medical device and should not be used as the sole basis for clinical decisions. Always verify AI-generated suggestions with clinical judgment and current medical guidelines.

## 🙏 Acknowledgments

- Built with [Anthropic Claude](https://www.anthropic.com/claude)
- OCR powered by [EasyOCR](https://github.com/JaidedAI/EasyOCR)
- BLE communication using [Bleak](https://github.com/hbldh/bleak)
- Computer vision using [OpenCV](https://opencv.org/)

## 📧 Contact

- Issues: [GitHub Issues](https://github.com/g150446/ehr-agentic-toolkit/issues)
- Email: your.email@example.com

---

**Made with ❤️ for healthcare professionals**
