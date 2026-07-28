import cv2
import numpy as np
import pytest

from automation.dual_cxr_crop import find_dual_cxr_boxes


def _dual_cxr_screen(active_side):
    image = np.zeros((1080, 1920, 3), dtype=np.uint8)
    gradient = np.linspace(30, 220, 640, dtype=np.uint8)[:, None]
    cxr = np.repeat(gradient, 680, axis=1)
    cxr = np.repeat(cxr[:, :, None], 3, axis=2)

    image[340:980, 130:810] = cxr
    image[340:980, 1090:1770] = cxr

    if active_side == "left":
        top_left, bottom_right = (10, 325), (950, 1015)
    else:
        top_left, bottom_right = (970, 325), (1910, 1015)
    cv2.rectangle(image, top_left, bottom_right, (220, 2, 2), 4)
    return image


@pytest.mark.parametrize("active_side", ["left", "right"])
def test_find_dual_cxr_boxes_supports_either_active_pane(active_side):
    boxes = find_dual_cxr_boxes(_dual_cxr_screen(active_side))

    assert boxes is not None
    left_box, right_box = boxes
    assert left_box[0] < right_box[0]
    assert left_box[2:] == (680, 640)
    assert right_box[2:] == (680, 640)
