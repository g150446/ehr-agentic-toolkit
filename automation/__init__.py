"""
Windows Login Automation Package

Automates Windows PC login using HDMI screen capture, OCR text extraction,
and ESP32 BLE keyboard/mouse control.
"""

from automation.config import AutomationConfig, load_config
from automation.ble_controller import BLEController
from automation.screen_analyzer import (
    capture_screen,
    load_ocr_reader,
    analyze_layout,
    extract_text,
    find_password_field,
    DetectedRegion,
    OCRResult,
    PasswordField
)
from automation.utils import (
    setup_logging,
    debug_pause,
    save_debug_image,
    draw_bounding_boxes,
    calculate_center,
    ProgressLogger
)

__version__ = "0.1.0"

__all__ = [
    # Config
    "AutomationConfig",
    "load_config",

    # BLE Controller
    "BLEController",

    # Screen Analysis
    "capture_screen",
    "load_ocr_reader",
    "analyze_layout",
    "extract_text",
    "find_password_field",
    "DetectedRegion",
    "OCRResult",
    "PasswordField",

    # Utilities
    "setup_logging",
    "debug_pause",
    "save_debug_image",
    "draw_bounding_boxes",
    "calculate_center",
    "ProgressLogger",
]
