"""Unit tests for automation.ollama_vlm_ime.

Tests cover:
- _ensure_min_size: small image upscaling to satisfy Qwen3VL SmartResize
- detect_patient_record_panel3: vertical divider detection for patient record screens
- crop_to_input_region: crops to 3rd panel when patient record screen is detected
"""

import numpy as np
import pytest
import cv2

from automation.ollama_vlm_ime import (
    _MIN_FRAME_HEIGHT,
    _MIN_FRAME_WIDTH,
    _ensure_min_size,
    detect_patient_record_panel3,
    crop_to_input_region,
)


def _make_frame(h: int, w: int) -> np.ndarray:
    """Create a dummy BGR frame of given size."""
    return np.zeros((h, w, 3), dtype=np.uint8)


class TestEnsureMinSize:
    def test_crash_case_58x25_is_upscaled(self):
        """The exact dimensions that caused the Qwen3VL crash must be upscaled."""
        frame = _make_frame(25, 58)
        result = _ensure_min_size(frame)
        h, w = result.shape[:2]
        assert h >= _MIN_FRAME_HEIGHT
        assert w >= _MIN_FRAME_WIDTH
        # Must satisfy Qwen3VL SmartResize factor:32
        assert h > 32
        assert w > 32

    def test_large_image_unchanged(self):
        """Images already larger than the minimum should not be resized."""
        frame = _make_frame(300, 400)
        result = _ensure_min_size(frame)
        assert result.shape[:2] == (300, 400)

    def test_exactly_at_minimum_unchanged(self):
        """Image at exactly the minimum dimensions should not be resized."""
        frame = _make_frame(_MIN_FRAME_HEIGHT, _MIN_FRAME_WIDTH)
        result = _ensure_min_size(frame)
        assert result.shape[:2] == (_MIN_FRAME_HEIGHT, _MIN_FRAME_WIDTH)

    def test_tiny_10x10_is_upscaled(self):
        """10×10 pixel image (noise-level ROI) must be upscaled."""
        frame = _make_frame(10, 10)
        result = _ensure_min_size(frame)
        h, w = result.shape[:2]
        assert h >= _MIN_FRAME_HEIGHT
        assert w >= _MIN_FRAME_WIDTH

    def test_aspect_ratio_preserved_wide_image(self):
        """Wide images should preserve aspect ratio (height scales proportionally)."""
        frame = _make_frame(20, 300)  # already wide enough but too short
        result = _ensure_min_size(frame)
        original_ratio = 300 / 20
        result_ratio = result.shape[1] / result.shape[0]
        assert abs(result_ratio - original_ratio) < 0.5  # within half-pixel rounding

    def test_aspect_ratio_preserved_tall_image(self):
        """Tall narrow images should preserve aspect ratio (width scales proportionally)."""
        frame = _make_frame(200, 10)  # tall but too narrow
        result = _ensure_min_size(frame)
        original_ratio = 10 / 200
        result_ratio = result.shape[1] / result.shape[0]
        assert abs(result_ratio - original_ratio) < 0.5

    def test_output_is_numpy_array(self):
        """Output must be a numpy ndarray (not a copy reference change)."""
        frame = _make_frame(25, 58)
        result = _ensure_min_size(frame)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.uint8

    def test_min_constants_above_qwen_factor(self):
        """MIN_FRAME constants must exceed Qwen3VL SmartResize factor:32."""
        assert _MIN_FRAME_HEIGHT > 32
        assert _MIN_FRAME_WIDTH > 32


def _make_patient_record_frame(
    h: int = 1080, w: int = 1920, divider_xs: list[int] | None = None
) -> np.ndarray:
    """Create a synthetic patient record screen with vertical dividers."""
    if divider_xs is None:
        divider_xs = [170, 720, 1440, 1720]
    frame = np.full((h, w, 3), 200, dtype=np.uint8)  # light gray background
    for x in divider_xs:
        # Draw a dark vertical line spanning full height
        cv2.line(frame, (x, 0), (x, h - 1), (60, 60, 60), 4)
    return frame


class TestDetectPatientRecordPanel3:
    def test_detects_panel3_in_synthetic_frame(self):
        """Synthetic frame with 4 clear dividers should return 3rd panel bounds."""
        dividers = [170, 720, 1440, 1720]
        frame = _make_patient_record_frame(divider_xs=dividers)
        result = detect_patient_record_panel3(frame)
        assert result is not None
        x1, x2 = result
        # Panel 3 is between divider[1] and divider[2] (within tolerance)
        assert abs(x1 - dividers[1]) < 50
        assert abs(x2 - dividers[2]) < 50

    def test_returns_none_for_blank_frame(self):
        """A plain uniform frame has no lines → should return None."""
        frame = np.full((1080, 1920, 3), 200, dtype=np.uint8)
        result = detect_patient_record_panel3(frame)
        assert result is None

    def test_returns_none_when_fewer_than_3_dividers(self):
        """Only 2 dividers (not a 4-panel layout) → should return None."""
        frame = _make_patient_record_frame(divider_xs=[400, 900])
        result = detect_patient_record_panel3(frame)
        assert result is None

    def test_panel3_x2_greater_than_x1(self):
        """Panel bounds must have x2 > x1 (non-empty region)."""
        frame = _make_patient_record_frame()
        result = detect_patient_record_panel3(frame)
        if result is not None:
            x1, x2 = result
            assert x2 > x1


class TestCropToInputRegion:
    def test_crops_to_panel3_on_patient_record_screen(self):
        """On a patient record screen, width should be reduced to panel 3."""
        dividers = [170, 720, 1440, 1720]
        frame = _make_patient_record_frame(divider_xs=dividers)
        result = crop_to_input_region(frame)
        full_w = frame.shape[1]
        cropped_w = result.shape[1]
        # Should be narrower than full frame
        assert cropped_w < full_w
        # Height unchanged
        assert result.shape[0] == frame.shape[0]

    def test_returns_full_frame_when_no_panel_detected(self):
        """When no panels detected, input region equals the original frame."""
        frame = np.full((1080, 1920, 3), 200, dtype=np.uint8)
        result = crop_to_input_region(frame)
        assert result.shape == frame.shape
