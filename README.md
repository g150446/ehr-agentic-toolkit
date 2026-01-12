# EHR AI Bridge Toolkit

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**AI-powered clinical decision support bridge for on-premises Electronic Health Record systems.**

EHR AI Bridge Toolkit connects your existing on-premises EHR system with AI capabilities without requiring direct system integration. Using screen capture and OCR technology, it extracts clinical information, anonymizes patient data, and provides AI-assisted clinical summaries, differential diagnoses, and treatment suggestions.

## ✨ Key Features

- 🏥 **Universal EHR Compatibility** - Works with any on-premises EHR via HDMI capture
- 🔒 **Privacy-First Design** - All processing happens locally, patient identifiers never stored
- 🤖 **AI-Assisted Clinical Support** - Summaries, differential diagnosis, treatment suggestions (planned)
- 🔌 **Plugin Architecture** - Easy to add support for new EHR systems
- 🛡️ **Enterprise-Grade Security** - Encrypted storage, audit logging, HIPAA-conscious design
- 🎯 **Zero EHR Modification** - No changes to existing systems required
- 📹 **Real-time Stream Monitor** - Debug HDMI capture with YOLO detection visualization ✅ **Implemented**
- 🎮 **Windows Automation** - Automated login via ESP32 BLE keyboard emulation ✅ **Implemented**
- 💬 **Interactive Browser Assistant** - Terminal chat for remote Chrome control with dual YOLO models (DocLayout + YOLOv8) ✅ **Implemented**

## 📊 Project Status

| Component | Status | Description |
|-----------|--------|-------------|
| **HDMI Capture** | ✅ **Complete** | Real-time video capture from MiraBox/compatible devices |
| **DocLayout-YOLO** | ✅ **Complete** | Document layout detection and UI element recognition |
| **OCR (EasyOCR)** | ✅ **Complete** | Multi-language text extraction (Japanese, English) |
| **Stream Monitor** | ✅ **Complete** | Interactive HDMI capture monitor with detection overlay |
| **Windows Automation** | ✅ **Complete** | Automated login via BLE keyboard emulation |
| **ESP32 BLE Control** | ✅ **Complete** | Keyboard/mouse HID emulation over Bluetooth |
| **Browser Assistant** | ✅ **Complete** | Interactive chat with DocLayout-YOLO + YOLOv8 UI detection |
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
git clone https://github.com/yourusername/ehr-ai-bridge-toolkit.git
cd ehr-ai-bridge-toolkit

# Run setup script (installs everything)
./scripts/setup_automation.sh

# Edit configuration
cp .env.example .env
nano .env  # Configure your settings
```

**Option 2: Manual Setup**
```bash
# Clone and enter directory
git clone https://github.com/yourusername/ehr-ai-bridge-toolkit.git
cd ehr-ai-bridge-toolkit

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install DocLayout-YOLO
cd DocLayout-YOLO
pip install -e .
cd ..

# Install EasyOCR
pip install easyocr

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

### Windows Login Automation

Automated Windows PC login using HDMI capture and ESP32 BLE keyboard emulation.

**Prerequisites:**
- Windows PC connected via HDMI to capture device
- ESP32 running BLE keyboard/mouse firmware
- MiraBox or compatible HDMI capture device

**Testing Components:**
```bash
# Test HDMI capture
./scripts/run_automation.sh --test-capture

# Test ESP32 BLE connection
./scripts/run_automation.sh --test-ble
```

**Run Automation:**
```bash
# Debug mode (step-by-step with pauses)
./scripts/run_automation.sh --password YOUR_PASSWORD --debug

# Automatic mode
./scripts/run_automation.sh --password YOUR_PASSWORD

# With live monitor window (NEW!)
./scripts/run_automation.sh --password YOUR_PASSWORD --monitor-mode

# Skip verification
./scripts/run_automation.sh --password YOUR_PASSWORD --no-verify
```

**How It Works:**
1. **Initialization** - Load YOLO model, OCR, connect to ESP32
2. **Screen Capture** - Capture Windows login screen via HDMI
3. **Screen Analysis** - Detect UI elements using DocLayout-YOLO + EasyOCR
4. **Input Control** - Send password via ESP32 BLE
5. **Verification** - Verify login success

---

### Interactive Browser Assistant

Terminal-based chat assistant for remote Chrome browser control using dual YOLO models.

#### Prerequisites

**Hardware:**
- Windows PC with Chrome installed
- HDMI connection from Windows PC to MiraBox capture device
- MiraBox (or compatible) HDMI capture device connected to Mac
- ESP32 module running BLE keyboard/mouse firmware
- ESP32 connected to Windows PC via USB (for HID emulation)

**Software:**
- Python virtual environment with dependencies installed
- Chrome browser on Windows PC
- ESP32 BLE device advertising as configured in `.env`

#### Quick Start

**1. First-time Setup**

```bash
# From project root
./scripts/setup_automation.sh

# Verify .env file has required settings
cat .env | grep -E "ESP32_DEVICE_NAME|BLE_"
```

**2. Start the Assistant**

```bash
# Basic mode
./scripts/run_browser_assistant.sh

# Debug mode (recommended for first time)
./scripts/run_browser_assistant.sh --debug

# Custom video device
./scripts/run_browser_assistant.sh --device 1 --debug
```

**3. Interactive Chat Session**

Once started, you'll see the chat prompt:

```
🤖 Interactive Browser Assistant
Control remote Chrome browser via chat commands.
Type 'help' for available commands.

[doclayout] >
```

The prompt shows the currently active YOLO model (`doclayout` or `ui-detection`).

#### Usage Examples

**Example 1: Open Chrome and Navigate**

```bash
[doclayout] > open chrome
✅ Chrome opened

[doclayout] > switch to ui detection
✅ Switched to ui-detection model

[ui-detection] > goto google.com
✅ Navigated to https://google.com
```

**Example 2: Analyze Screen with Different Models**

```bash
# With DocLayout-YOLO (documents)
[doclayout] > analyze
📊 Detected 5 elements:
  1. text (confidence: 0.92)
  2. title (confidence: 0.88)
  3. figure (confidence: 0.85)
  ...

# Switch to UI detection model (web forms)
[doclayout] > switch to ui detection
✅ Switched to ui-detection model

# Analyze again with UI model
[ui-detection] > analyze
📊 Detected 8 elements:
  1. text (confidence: 0.85)
  2. button (confidence: 0.78)
  3. input (confidence: 0.92)
  ...
```

**Example 3: Navigate Multiple Sites**

```bash
[ui-detection] > goto github.com
✅ Navigated to https://github.com

[ui-detection] > goto stackoverflow.com
✅ Navigated to https://stackoverflow.com

[ui-detection] > goto https://www.wikipedia.org
✅ Navigated to https://www.wikipedia.org
```

**Example 4: Capture Screenshots**

```bash
[ui-detection] > capture
✅ Saved: automation_outputs/chat_screenshots/chat_capture_20260112_170530.jpg

[ui-detection] > analyze
📊 Detected 12 elements:
  ...

[ui-detection] > capture
✅ Saved: automation_outputs/chat_screenshots/chat_capture_20260112_170545.jpg
```

#### Available Commands

| Command | Description | Example |
|---------|-------------|---------|
| **Browser Control** | | |
| `open chrome` | Launch Chrome via Windows search | `open chrome` |
| `goto <url>` | Navigate to URL (with or without https://) | `goto google.com` or `goto https://example.com` |
| `click address bar` | Manually click address bar | `click address bar` |
| **Model Control** | | |
| `switch to doclayout` | Use DocLayout-YOLO (document analysis) | `switch to doclayout` |
| `switch to ui detection` | Use YOLOv11 UI model (web forms) | `switch to ui detection` |
| `use ui model` | Alternative way to switch to UI model | `use ui model` |
| **Screen Analysis** | | |
| `analyze` | Analyze current screen with active model | `analyze` |
| `capture` | Save screenshot of current screen | `capture` |
| **Utility** | | |
| `help` | Show all available commands | `help` |
| `quit` or `exit` | Exit the assistant | `quit` |

#### How It Works

**Opening Chrome:**
1. Presses Windows key (opens Start menu)
2. Types "chrome" (Windows search finds Chrome)
3. Presses Enter (launches first result)

**Navigating to URL:**
1. Switches to UI detection model temporarily
2. Captures screen via HDMI
3. Detects address bar location (looks for text input in top 30% of screen)
4. Moves mouse to address bar center
5. Clicks address bar
6. Clears existing content (Ctrl+A, Delete)
7. Types URL
8. Presses Enter
9. Restores previous YOLO model

**Model Switching:**
- **DocLayout-YOLO**: Best for document layout (tables, text blocks, figures, titles)
- **YOLOv8 UI Detection** (`foduucom/web-form-ui-field-detection`): Best for web form UI elements (text inputs, buttons, radio buttons, checkboxes, search bars)
- Models are lazy-loaded (only loaded when first used)
- Switching is instant after initial load
- UI model detects: Name fields, Email fields, Password fields, Buttons, Radio buttons, Checkboxes, Text inputs

#### Output Files

**Screenshots:**
```
automation_outputs/chat_screenshots/
├── chat_capture_20260112_170530.jpg
├── chat_capture_20260112_170545.jpg
└── ...
```

**Logs:**
```
automation_outputs/logs/
└── browser_assistant_20260112_170000.log
```

#### Command-Line Options

```bash
python -m automation.browser_assistant \
  --device 0                      # Video capture device index (default: 0)
  --env-file .env                 # Path to .env file (default: .env)
  --output-dir ./screenshots      # Screenshot directory (default: automation_outputs/chat_screenshots)
  --debug                         # Enable debug logging
```

#### Troubleshooting

**"Failed to load YOLOv8 UI detection model"**

Install ultralyticsplus (required for web form UI detection):
```bash
source venv/bin/activate
pip install ultralyticsplus>=0.0.28 ultralytics>=8.0.0
```

If you get an import error, the package might not be installed. Run setup again:
```bash
./scripts/setup_automation.sh
```

**"Address bar not detected"**

- Make sure Chrome is open and visible
- Try switching to UI detection model first: `switch to ui detection`
- Ensure the address bar is in the top 30% of the screen
- Check HDMI capture is showing Chrome window (not minimized)

**"BLE connection failed"**

- Check ESP32 is powered on and advertising
- Verify Bluetooth is enabled on Mac
- Check device name in `.env` matches ESP32's advertised name:
  ```bash
  grep ESP32_DEVICE_NAME .env
  ```
- Test BLE connection: `./scripts/run_automation.sh --test-ble`

**"Failed to capture screen"**

- Check MiraBox is connected and powered
- Verify Windows PC HDMI is connected to MiraBox
- Try different device index: `./scripts/run_browser_assistant.sh --device 1`
- Test capture: `./scripts/run_automation.sh --test-capture`

**Chrome doesn't open**

- Ensure Chrome is installed on Windows PC
- Try typing "chrome" manually in Windows search to verify it appears
- Check ESP32 BLE connection is working
- Try running in debug mode to see detailed logs

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
cd /path/to/ehr-ai-bridge-toolkit
./scripts/run_monitor.sh
```

Or set PYTHONPATH manually:
```bash
export PYTHONPATH=/path/to/ehr-ai-bridge-toolkit:$PYTHONPATH
python -m automation.monitor_stream
```

#### "No module named 'doclayout_yolo'"

**Solution:** Install DocLayout-YOLO in your virtual environment:
```bash
source venv/bin/activate
cd DocLayout-YOLO
pip install -e .
cd ..
```

Or run the setup script:
```bash
./scripts/setup_automation.sh
```

#### "Configuration error: DocLayout-YOLO model not found"

**Solution:** Download the pre-trained model:
```bash
source venv/bin/activate
python3 -c "
from doclayout_yolo import YOLOv10
model = YOLOv10.from_pretrained('juliozhao/DocLayout-YOLO-DocStructBench')
model.save('DocLayout-YOLO/models/doclayout.pt')
"
```

#### ESP32 Not Connecting

**Checklist:**
1. ✅ Bluetooth enabled on Mac/PC
2. ✅ ESP32 powered and advertising
3. ✅ Device name in `.env` matches ESP32's advertised name
4. ✅ Test connection: `./scripts/run_automation.sh --test-ble`

#### Screen Capture Not Working

**Checklist:**
1. ✅ HDMI capture device (MiraBox) connected and powered
2. ✅ Windows PC HDMI output connected to capture device
3. ✅ Correct device index (usually 0, try 1-2 if issues)
4. ✅ Test capture: `./scripts/run_automation.sh --test-capture`

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
- **Issues:** Report bugs at [GitHub Issues](https://github.com/yourusername/ehr-ai-bridge-toolkit/issues)
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
- Document layout analysis powered by [DocLayout-YOLO](https://github.com/opendatalab/DocLayout-YOLO)
- OCR powered by [EasyOCR](https://github.com/JaidedAI/EasyOCR)
- BLE communication using [Bleak](https://github.com/hbldh/bleak)
- Computer vision using [OpenCV](https://opencv.org/)

## 📧 Contact

- Issues: [GitHub Issues](https://github.com/yourusername/ehr-ai-bridge-toolkit/issues)
- Email: your.email@example.com

---

**Made with ❤️ for healthcare professionals**
