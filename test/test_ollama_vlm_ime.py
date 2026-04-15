"""Unit tests for automation.ollama_vlm_ime._ensure_min_size.

Verifies that small images are upscaled to satisfy Qwen3VL's SmartResize
requirement (both dimensions > factor:32) before being sent to the model runner.
"""

import numpy as np
import pytest

from automation.ollama_vlm_ime import _MIN_FRAME_HEIGHT, _MIN_FRAME_WIDTH, _ensure_min_size


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
