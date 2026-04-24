import numpy as np

from automation.history_panel_analyzer import _extract_date_candidates, infer_history_panel_roi


def test_extract_date_candidates_marks_exact_match():
    ocr_results = [
        (
            [[289, 643], [379, 643], [379, 665], [289, 665]],
            "2026年日4月03日15:日1",
            0.93,
        ),
        (
            [[1080, 160], [1182, 160], [1182, 184], [1080, 184]],
            "メニュー",
            0.91,
        ),
    ]

    candidates = _extract_date_candidates(
        ocr_results,
        image_width=1920,
        target_date="20260403",
    )

    assert len(candidates) == 1
    assert candidates[0]["exact_match"] is True
    assert candidates[0]["cx"] == 334


def test_infer_history_panel_roi_prefers_dense_vertical_cluster():
    image = np.zeros((1080, 1920, 3), dtype=np.uint8)
    candidates = [
        {
            "bbox": {"x1": 280, "y1": 640, "x2": 380, "y2": 664},
            "cx": 330,
            "cy": 652,
            "width": 100,
            "height": 24,
            "exact_match": False,
        },
        {
            "bbox": {"x1": 282, "y1": 720, "x2": 382, "y2": 744},
            "cx": 332,
            "cy": 732,
            "width": 100,
            "height": 24,
            "exact_match": True,
        },
        {
            "bbox": {"x1": 281, "y1": 900, "x2": 381, "y2": 924},
            "cx": 331,
            "cy": 912,
            "width": 100,
            "height": 24,
            "exact_match": False,
        },
        {
            "bbox": {"x1": 1080, "y1": 160, "x2": 1200, "y2": 188},
            "cx": 1140,
            "cy": 174,
            "width": 120,
            "height": 28,
            "exact_match": False,
        },
    ]

    roi = infer_history_panel_roi(candidates, image_shape=image.shape)

    assert roi is not None
    assert roi["x1"] < 300
    assert roi["x2"] > 370
    assert roi["y1"] < 650
    assert roi["y2"] > 910
    assert roi["candidate_count"] == 3
