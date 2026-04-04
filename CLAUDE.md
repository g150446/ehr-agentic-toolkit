# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

EHR AI Bridge Toolkit is an AI-powered clinical decision support bridge for on-premises Electronic Health Record systems. It uses screen capture and OCR technology to extract clinical information from EHR displays, anonymizes patient data, and provides AI-assisted clinical summaries and decision support.

The project integrates **DocLayout-YOLO**, a document layout analysis model, for detecting and extracting structured regions from EHR screen captures.

## Architecture

The system follows a layered architecture:

1. **Capture Layer**: HDMI screen capture from on-premises EHR systems
2. **Document Layout Detection**: DocLayout-YOLO identifies UI regions and document structure
3. **OCR Layer**: EasyOCR extracts text from detected regions (supports Japanese and English)
4. **Adapter Layer**: EHR-specific parsers convert raw OCR to structured data
5. **Anonymization Layer**: Removes PHI (patient identifiable information)
6. **Storage Layer**: AES-256 encrypted PostgreSQL database on external SSD
7. **AI Engine**: Claude API or local LLM for clinical decision support

### Key Design Principles

- **Privacy-First**: No PHI storage, local processing capability
- **Universal Compatibility**: Works via screen capture, no EHR modification required
- **Plugin Architecture**: Custom EHR adapters in `ehr_ai_bridge/adapters/`

## Python Environment Setup

**CRITICAL: Always use virtual environment (venv) for Python development in this project.**

### Initial Setup

```bash
# Create virtual environment in project root
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # On macOS/Linux
# OR
venv\Scripts\activate  # On Windows

# Upgrade pip
pip install --upgrade pip

# Install project dependencies
pip install -r requirements.txt

# Install DocLayout-YOLO in editable mode
cd DocLayout-YOLO
pip install -e .
cd ..

# Verify PYTHONPATH for automation module
export PYTHONPATH=/path/to/ehr-agentic-toolkit:$PYTHONPATH
```

### Why Virtual Environment?

1. **Dependency Isolation**: Prevents conflicts between project dependencies and system packages
2. **Reproducibility**: Ensures consistent environment across development machines
3. **Clean Uninstall**: Easy to remove all project dependencies by deleting venv directory
4. **Version Control**: Each project can use different versions of the same library

### For Claude Code

When installing Python libraries, **ALWAYS**:
1. Check if virtual environment is activated (look for `(venv)` in prompt)
2. If not activated, activate it first: `source venv/bin/activate`
3. Install packages using `pip install` (will install to venv, not system-wide)
4. Never install packages globally using system pip unless explicitly requested

### Quick Check

```bash
# Verify you're using venv Python
which python  # Should show: /path/to/project/venv/bin/python

# Check installed packages in venv
pip list
```

## Development Commands

### DocLayout-YOLO (Document Layout Analysis)

The DocLayout-YOLO component is located in the `DocLayout-YOLO/` subdirectory and is used for detecting document regions in captured EHR screens.

```bash
# Navigate to DocLayout-YOLO directory
cd DocLayout-YOLO

# Setup environment (first time)
conda create -n doclayout_yolo python=3.10
conda activate doclayout_yolo
pip install -e .

# Or install just for inference
pip install doclayout-yolo

# Download pre-trained model
python download_model.py

# Run inference on a single image
python demo.py --model path/to/model --image-path path/to/image --imgsz 1024 --conf 0.2

# Capture screen from HDMI device
python capture_windows.py

# Test with OCR extraction
python test_with_ocr.py

# Test with position information
python test_with_ocr_positions.py

# Train model
python train.py --data <dataset> --model m-doclayout --epoch 500 --image-size 1600 --batch-size 64 --project <output_dir> --optimizer SGD --lr0 0.04

# Evaluate model
python val.py --data <dataset> --model checkpoint.pt --device 0 --batch-size 64

# Format DocSynth300K dataset for training
python format_docsynth300k.py
```

### ESP32 BLE and HDMI Capture Testing

```bash
# Test BLE connection to ESP32
./scripts/run_ble_test.sh

# Test HDMI screen capture (use monitor script)
./scripts/run_monitor.sh
```

**Configuration**: Edit `.env` file in project root:
- `ESP32_DEVICE_NAME`: BLE device name (default: "BLE Mouse & Keyboard")
- `CAPTURE_DEVICE_INDEX`: Video capture device index (default: 0)

### Main EHR Bridge Commands

```bash
# Setup encrypted external SSD
./scripts/setup_ssd.sh

# Initialize database
./scripts/start_postgresql.sh
ehr-bridge init

# Configure EHR system (e.g., Fujitsu)
ehr-bridge configure --ehr-type fujitsu

# Start the bridge
ehr-bridge start

# Interactive mode
ehr-bridge start --interactive

# Test EHR connection
ehr-bridge test-ehr

# View logs
ehr-bridge logs --tail 50
```

### Testing and Development

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run tests with coverage
pytest --cov

# Format code (uses black)
black ehr_ai_bridge/

# Lint code (uses ruff)
ruff check ehr_ai_bridge/

# Type checking
mypy ehr_ai_bridge/
```

## Code Structure

### DocLayout-YOLO Module

- `DocLayout-YOLO/doclayout_yolo/`: Core YOLO implementation
  - `models/`: YOLO model architectures (v6, v8, v10, FastSAM, RT-DETR)
  - `nn/`: Neural network modules and layers
  - `cfg/`: Configuration files for datasets and models
  - `utils/`: Utility functions and callbacks
  - `trackers/`: Object tracking implementations
  - `solutions/`: Pre-built solutions for common tasks

- `DocLayout-YOLO/mesh-candidate_bestfit/`: Synthetic data generation pipeline
  - Implements "Mesh-candidate BestFit" algorithm for document synthesis
  - Used to create the DocSynth300K dataset

- Key scripts:
  - `demo.py`: Basic inference demo
  - `train.py`: Model training with extensive hyperparameter control
  - `val.py`: Model validation/evaluation
  - `capture_windows.py`: Captures frames from HDMI video capture device (MiraBox)
  - `test_with_ocr.py`: Runs layout detection + OCR extraction
  - `test_with_ocr_positions.py`: Adds position/sorting information to OCR output

### EHR Bridge Module (Planned)

- `ehr_ai_bridge/adapters/`: EHR-specific parsers
  - `fujitsu/`: Fujitsu EHR adapter
  - `base.py`: BaseEHRAdapter interface

- Core layers (planned):
  - Capture layer: Screen capture integration
  - Anonymization: PHI removal utilities
  - AI engine: LLM integration for clinical support

## Device and Hardware Notes

### Video Capture

The system uses **MiraBox Video Capture** device (device index 0) to capture HDMI output from Windows PCs running EHR systems. The `capture_windows.py` script demonstrates basic frame capture from this device.

### Accelerator Support

DocLayout-YOLO automatically selects the best available device:
- CUDA (NVIDIA GPU)
- MPS (Apple Silicon)
- CPU (fallback)

For OCR, EasyOCR can use GPU acceleration if available (set `gpu=True` in `easyocr.Reader()`).

## Dataset Configuration

Dataset YAML files are located in `DocLayout-YOLO/doclayout_yolo/cfg/datasets/`:
- `docsynth300k.yaml`: Large-scale synthetic pre-training dataset
- `d4la.yaml`: D4LA benchmark dataset
- `doclaynet.yaml`: DocLayNet benchmark dataset

Data should be placed in `./layout_data/` with the following structure:
```
./layout_data
├── D4LA/
│   ├── images/
│   ├── labels/
│   ├── test.txt
│   └── train.txt
├── doclaynet/
│   ├── images/
│   ├── labels/
│   ├── val.txt
│   └── train.txt
└── docsynth300k/
    └── (formatted from .parquet files)
```

## OCR Pipeline Integration

The project uses **EasyOCR** for text extraction with Japanese and English language support:

1. DocLayout-YOLO detects document regions (tables, text blocks, figures, etc.)
2. Each region is cropped and saved
3. EasyOCR extracts text from each cropped region
4. Results include both layout structure and extracted text with confidence scores
5. Regions are sorted spatially (top-to-bottom, left-to-right) for logical reading order

This approach is optimized for Japanese EHR systems which often contain mixed Japanese/English text.

## Creating Custom EHR Adapters

To add support for a new EHR system:

1. Create directory: `ehr_ai_bridge/adapters/your_ehr/`
2. Implement `BaseEHRAdapter` interface
3. Add YAML configuration for layout patterns
4. Define region-to-field mappings specific to your EHR's UI
5. Write tests for the adapter
6. Submit PR with documentation

Reference the Fujitsu adapter for implementation patterns.

## Security Considerations

- Patient identifiers must never be persisted to disk
- All database storage uses AES-256 encryption on external SSD
- Audit logging captures all system operations
- Anonymization uses age ranges and gender only
- Support for fully offline/air-gapped deployment

## Model Downloads

Pre-trained DocLayout-YOLO models are available on HuggingFace:
- DocStructBench fine-tuned model (general purpose): `juliozhao/DocLayout-YOLO-DocStructBench`
- DocSynth300K pre-trained model: `juliozhao/DocLayout-YOLO-DocSynth300K-pretrain`
- D4LA models: `juliozhao/DocLayout-YOLO-D4LA-*`
- DocLayNet models: `juliozhao/DocLayout-YOLO-DocLayNet-*`

Load models using:
```python
from doclayout_yolo import YOLOv10
model = YOLOv10.from_pretrained("juliozhao/DocLayout-YOLO-DocStructBench")
```

## Troubleshooting

### Memory Issues During Training

Due to memory leakage in YOLO's data loading code, large-scale pretraining may be interrupted. Use `--pretrain last_checkpoint.pt --resume` to resume from checkpoint.

### SSL Certificate Issues

If EasyOCR model download fails due to SSL, add:
```python
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
```

### Video Capture Device Not Found

Verify MiraBox device connection with `ls /dev/video*` (Linux) or check device index with OpenCV's `cv2.VideoCapture()` test.
