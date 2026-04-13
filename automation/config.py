"""
Configuration module for Windows login automation.

Loads settings from environment variables (.env file) and provides
configuration management for the automation system.
"""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv


class AutomationConfig:
    """Configuration manager for Windows login automation."""

    def __init__(self, env_file: Optional[str] = None):
        """
        Initialize configuration from environment file.

        Args:
            env_file: Path to .env file. If None, searches in standard locations.
        """
        # Load environment variables
        if env_file:
            load_dotenv(env_file, override=True)
        else:
            # Search for .env in project root
            project_root = Path(__file__).parent.parent
            env_path = project_root / '.env'
            if env_path.exists():
                load_dotenv(env_path, override=True)

        # ESP32 BLE Configuration
        self.esp32_device_name = os.getenv('ESP32_DEVICE_NAME', 'BLE Mouse & Keyboard')
        self.ble_service_uuid = os.getenv('BLE_SERVICE_UUID', '6E400001-B5A3-F393-E0A9-E50E24DCCA9E')
        self.ble_rx_char_uuid = os.getenv('BLE_RX_CHAR_UUID', '6E400002-B5A3-F393-E0A9-E50E24DCCA9E')
        self.ble_tx_char_uuid = os.getenv('BLE_TX_CHAR_UUID', '6E400003-B5A3-F393-E0A9-E50E24DCCA9E')

        # Windows Login Credentials
        self.password = os.getenv('WINDOWS_LOGIN_PASSWORD', '')

        # Login Automation Settings
        self.debug_mode = os.getenv('LOGIN_DEBUG_MODE', 'true').lower() == 'true'
        self.auto_verify = os.getenv('LOGIN_AUTO_VERIFY', 'true').lower() == 'true'
        self.retry_count = int(os.getenv('LOGIN_RETRY_COUNT', '3'))
        self.retry_delay = float(os.getenv('LOGIN_RETRY_DELAY', '1.0'))

        # Video Capture Configuration
        self.capture_device_index = int(os.getenv('CAPTURE_DEVICE_INDEX', '0'))
        self.capture_width = int(os.getenv('CAPTURE_WIDTH', '1920'))
        self.capture_height = int(os.getenv('CAPTURE_HEIGHT', '1080'))

        # Detection Configuration (used by UI detection model and analysis helpers)
        self.detection_confidence = float(os.getenv('DETECTION_CONFIDENCE', '0.2'))
        self.detection_image_size = int(os.getenv('DETECTION_IMAGE_SIZE', '1024'))
        self.detection_device = os.getenv('DETECTION_DEVICE', 'auto')

        # OCR Configuration
        ocr_languages = os.getenv('OCR_LANGUAGES', 'ja,en')
        self.ocr_languages = [lang.strip() for lang in ocr_languages.split(',')]
        # Default backend: PaddleOCR. Use PP-OCRv4 where supported; Japanese falls back to PP-OCRv5.
        self.ocr_backend = os.getenv('OCR_BACKEND', 'paddleocr')
        # Detection mode: 'yolo' (UI element detection first, then per-element OCR) or 'ocr' (full-image OCR only)
        # yolo mode is more reliable for menus/tab bars where OCR merges adjacent items into one segment.
        self.detection_mode = os.getenv('DETECTION_MODE', 'yolo')
        # For EasyOCR only: auto-enable MPS on Apple Silicon if not overridden via env var
        try:
            import torch
            _mps_available = torch.backends.mps.is_available()
        except Exception:
            _mps_available = False
        self.ocr_use_gpu = os.getenv('OCR_USE_GPU', 'true' if _mps_available else 'false').lower() == 'true'
        self.ocr_server_socket_path = os.getenv('OCR_SERVER_SOCKET_PATH', '/tmp/paddle_ocr_server.sock')
        self.ocr_server_timeout = float(os.getenv('OCR_SERVER_TIMEOUT', '120'))
        self.ocr_server_device = os.getenv('OCR_SERVER_DEVICE', 'auto')

        # Output Paths
        self.output_dir = Path(os.getenv('LOGIN_OUTPUT_DIR', './automation_outputs'))
        self.log_dir = self.output_dir / 'logs'
        self.screenshot_dir = self.output_dir / 'screenshots'

        # Ensure output directories exist
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)

        # Logging Configuration
        self.log_level = os.getenv('LOG_LEVEL', 'INFO')

    def validate(self, skip_password: bool = False) -> tuple[bool, list[str]]:
        """
        Validate configuration settings.

        Args:
            skip_password: If True, skip password validation (for image analysis mode)

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        if not skip_password and not self.password:
            errors.append("WINDOWS_LOGIN_PASSWORD is not set")

        if self.retry_count < 0:
            errors.append(f"LOGIN_RETRY_COUNT must be non-negative: {self.retry_count}")

        if self.retry_delay < 0:
            errors.append(f"LOGIN_RETRY_DELAY must be non-negative: {self.retry_delay}")

        if self.detection_confidence < 0 or self.detection_confidence > 1:
            errors.append(f"DETECTION_CONFIDENCE must be between 0 and 1: {self.detection_confidence}")

        return (len(errors) == 0, errors)

    def __repr__(self) -> str:
        """String representation of configuration."""
        return (
            f"AutomationConfig(\n"
            f"  esp32_device_name={self.esp32_device_name},\n"
            f"  password={'*' * len(self.password) if self.password else 'NOT SET'},\n"
            f"  debug_mode={self.debug_mode},\n"
            f"  capture_device={self.capture_device_index},\n"
            f"  ocr_backend={self.ocr_backend},\n"
            f"  ocr_server_socket_path={self.ocr_server_socket_path},\n"
            f"  output_dir={self.output_dir}\n"
            f")"
        )


def load_config(env_file: Optional[str] = None, skip_password: bool = False) -> AutomationConfig:
    """
    Load and validate configuration.

    Args:
        env_file: Path to .env file. If None, searches in standard locations.
        skip_password: If True, skip password validation (for image analysis mode)

    Returns:
        AutomationConfig instance

    Raises:
        ValueError: If configuration validation fails
    """
    config = AutomationConfig(env_file)
    is_valid, errors = config.validate(skip_password=skip_password)

    if not is_valid:
        error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        raise ValueError(error_msg)

    return config
