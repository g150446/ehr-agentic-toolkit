# EHR AI Bridge Toolkit

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**AI-powered clinical decision support bridge for on-premises Electronic Health Record systems.**

EHR AI Bridge Toolkit connects your existing on-premises EHR system with AI capabilities without requiring direct system integration. Using screen capture and OCR technology, it extracts clinical information, anonymizes patient data, and provides AI-assisted clinical summaries, differential diagnoses, and treatment suggestions.

## ✨ Key Features

- 🏥 **Universal EHR Compatibility** - Works with any on-premises EHR via HDMI capture
- 🔒 **Privacy-First Design** - All processing happens locally, patient identifiers never stored
- 🤖 **AI-Assisted Clinical Support** - Summaries, differential diagnosis, treatment suggestions
- 🔌 **Plugin Architecture** - Easy to add support for new EHR systems
- 🛡️ **Enterprise-Grade Security** - Encrypted storage, audit logging, HIPAA-conscious design
- 🎯 **Zero EHR Modification** - No changes to existing systems required

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

- macOS 11+ (M1 or later recommended)
- Python 3.10+
- HDMI capture device
- External SSD (for encrypted storage)
- PostgreSQL 15+

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/ehr-ai-bridge-toolkit.git
cd ehr-ai-bridge-toolkit

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install package
pip install -e .

# Or install with all optional dependencies
pip install -e ".[all]"
```

### Initial Setup
```bash
# 1. Set up encrypted external SSD
./scripts/setup_ssd.sh

# 2. Initialize database
./scripts/start_postgresql.sh
ehr-bridge init

# 3. Configure your EHR system
ehr-bridge configure --ehr-type fujitsu

# 4. Set up environment variables
cp .env.example .env
# Edit .env with your settings
```

### Usage
```bash
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

- [Getting Started Guide](docs/getting-started.md)
- [Architecture Overview](docs/architecture.md)
- [Security Guidelines](docs/security-guidelines.md)
- [EHR Configuration Guide](docs/ehr-configuration-guide.md)
- [API Reference](docs/api-reference.md)
- [Custom Adapter Development](docs/custom-ehr-setup.md)

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
- OCR powered by [Tesseract](https://github.com/tesseract-ocr/tesseract)
- UI detection using [YOLO](https://github.com/ultralytics/ultralytics)

## 📧 Contact

- Issues: [GitHub Issues](https://github.com/yourusername/ehr-ai-bridge-toolkit/issues)
- Email: your.email@example.com

---

**Made with ❤️ for healthcare professionals**
