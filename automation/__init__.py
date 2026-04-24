"""
Windows Login Automation Package

Automates Windows PC login using HDMI screen capture, OCR text extraction,
and ESP32 BLE keyboard/mouse control.
"""

from __future__ import annotations

from importlib import import_module

__version__ = "0.1.0"

_EXPORT_MAP = {
    "AutomationConfig": ("automation.config", "AutomationConfig"),
    "load_config": ("automation.config", "load_config"),
    "BLEController": ("automation.ble_controller", "BLEController"),
    "capture_screen": ("automation.screen_analyzer", "capture_screen"),
    "load_ocr_reader": ("automation.screen_analyzer", "load_ocr_reader"),
    "analyze_layout": ("automation.screen_analyzer", "analyze_layout"),
    "extract_text": ("automation.screen_analyzer", "extract_text"),
    "find_password_field": ("automation.screen_analyzer", "find_password_field"),
    "DetectedRegion": ("automation.screen_analyzer", "DetectedRegion"),
    "OCRResult": ("automation.screen_analyzer", "OCRResult"),
    "PasswordField": ("automation.screen_analyzer", "PasswordField"),
    "setup_logging": ("automation.utils", "setup_logging"),
    "debug_pause": ("automation.utils", "debug_pause"),
    "save_debug_image": ("automation.utils", "save_debug_image"),
    "draw_bounding_boxes": ("automation.utils", "draw_bounding_boxes"),
    "calculate_center": ("automation.utils", "calculate_center"),
    "ProgressLogger": ("automation.utils", "ProgressLogger"),
}

__all__ = list(_EXPORT_MAP)


def __getattr__(name: str):
    try:
        module_name, attr_name = _EXPORT_MAP[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
