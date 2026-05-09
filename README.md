# EHR Agentic Toolkit

![Physical Setup](hardware-agent/images/relationships.png)

**AI-powered clinical documentation automation toolkit for Electronic Health Record systems.**

EHR Agentic Toolkit connects your existing EHR system with AI capabilities without requiring direct system integration. Whether your EHR is on-premises or browser-based, the toolkit uses screen capture and OCR technology to extract clinical information and automate clinical documentation generation.

**Current capabilities:**
- Automated creation of medical referral letters (診療情報提供書)
- Automated discharge summary generation (退院時サマリ)
- AI-assisted text input via IME conversion for existing EHR fields

**Future roadmap:**
- Differential diagnosis assistance (鑑別診断支援)
- Treatment suggestion support
- Broader clinical decision support features

## Components

| Component | Description | Location |
|---|---|---|
| **Hardware Agent** | Python automation pipeline for on-premises EHR: HDMI capture, OCR, AI-assisted text input, discharge summary generation | [`hardware-agent/`](hardware-agent/README.md) |
| **EHR-Agent (Swift)** | macOS native AI chat application with screen capture and debugging for browser-based EHR access | [`swift-appkit/`](swift-appkit/README.md) |

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Disclaimer

This software is provided for research and development purposes. It is not a medical device and should not be used as the sole basis for clinical decisions. Always verify AI-generated suggestions with clinical judgment and current medical guidelines.

## Acknowledgments

- Built with [Anthropic Claude](https://www.anthropic.com/claude)
- OCR powered by [EasyOCR](https://github.com/JaidedAI/EasyOCR)
- BLE communication using [Bleak](https://github.com/hbldh/bleak)
- Computer vision using [OpenCV](https://opencv.org/)
