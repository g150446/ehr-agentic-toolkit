"""
Dual CXR Crop

Crops two chest X-ray (CXR) regions out of an HDMI capture of the PSP Viewer
reading screen (読影画面) when it is showing two panes side by side (e.g. a
current-vs-prior comparison layout), and blacks out the viewer's burned-in
annotations in each crop.

Detection reuses the primitives from automation.cxr_crop:

  1. Find the blue selection frame the viewer draws around the active pane
     -- this brackets the vertical band the images sit in.
  2. Find the radiograph inside that frame.
  3. Search the region on the opposite side of the frame, restricted to the
     same vertical band, for the second radiograph.
     Restricting to that band is what keeps the second pane's bright blob
     from merging with the toolbar/thumbnail strip above it.

This only supports a horizontal (left/right) 2-pane layout. It cannot find
the panes if no pane is selected (no blue frame present at all) -- use
--roi1/--roi2 to specify both regions manually in that case.

Usage:
    # Crop both CXRs from an image file
    python -m automation.dual_cxr_crop --image captures/two-xray.jpg --out1 /tmp/left.png --out2 /tmp/right.png

    # Crop from a live HDMI capture
    python -m automation.dual_cxr_crop --device 0 --out1 /tmp/left.png --out2 /tmp/right.png

    # Bypass auto-detection
    python -m automation.dual_cxr_crop --image captures/two-xray.jpg \\
        --roi1 132,339,692,665 --roi2 1028,344,692,660 --out1 /tmp/left.png --out2 /tmp/right.png
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import cv2

from automation.config import load_config
from automation.utils import setup_logging, save_debug_image
from automation.cxr_crop import (
    Box,
    CxrCropResult,
    _parse_roi,
    crop_cxr,
    find_cxr_region,
    find_viewer_viewport,
)

logger = logging.getLogger("dual_cxr_crop")


def find_dual_cxr_boxes(
    image,
    debug_dir: Optional[Path] = None,
) -> Optional[Tuple[Box, Box]]:
    """
    Locate two side-by-side radiographs inside a screen capture.

    Args:
        image: Full screen capture (BGR)
        debug_dir: If set, intermediate blob masks are saved here

    Returns:
        (left_box, right_box) in full-image coordinates, or None if the
        active pane's frame (needed to bracket the vertical band) or either
        radiograph could not be found.
    """
    full_w = image.shape[1]

    viewport = find_viewer_viewport(image)
    if viewport is None:
        logger.debug("No active pane frame found; cannot bracket the dual-pane layout")
        return None

    vx, vy, vw, vh = viewport

    active_box = find_cxr_region(image, viewport, debug_dir)
    if active_box is None:
        logger.debug("No CXR found in the active pane")
        return None

    # The active pane can be either the left or right image. Search the
    # opposite side while keeping the vertical band established by the frame.
    frame_center_x = vx + vw / 2
    if frame_center_x < full_w / 2:
        opposite_x = vx + vw
        opposite_w = full_w - opposite_x
        active_side = "left"
    else:
        opposite_x = 0
        opposite_w = vx
        active_side = "right"

    if opposite_w <= 0:
        logger.debug(
            f"Active {active_side} pane frame leaves no room for the opposite pane"
        )
        return None

    # There is no frame on the opposite pane, so this is a synthetic viewport.
    # The small inset find_cxr_region applies is harmless on a borderless canvas.
    opposite_viewport = (opposite_x, vy, opposite_w, vh)
    opposite_box = find_cxr_region(image, opposite_viewport, debug_dir)
    if opposite_box is None:
        logger.debug(f"No CXR found opposite the active {active_side} pane")
        return None

    left_box, right_box = sorted((active_box, opposite_box), key=lambda box: box[0])
    logger.debug(f"Dual CXR boxes: left={left_box} right={right_box}")
    return left_box, right_box


def crop_dual_cxr(
    image,
    roi1: Optional[Box] = None,
    roi2: Optional[Box] = None,
    mask_overlay: bool = True,
    debug_dir: Optional[Path] = None,
) -> Optional[Tuple[CxrCropResult, CxrCropResult]]:
    """
    Crop two chest X-ray regions out of a screen capture.

    Args:
        image: Full screen capture (BGR)
        roi1: Manual (x, y, w, h) for the first CXR. Bypasses auto-detection.
        roi2: Manual (x, y, w, h) for the second CXR. Bypasses auto-detection.
        mask_overlay: Black out viewer annotations inside each crop
        debug_dir: If set, intermediate images are saved here

    Returns:
        (result1, result2), or None if either CXR could not be found/cropped
    """
    if roi1 is not None and roi2 is not None:
        box1, box2 = roi1, roi2
    else:
        boxes = find_dual_cxr_boxes(image, debug_dir)
        if boxes is None:
            return None
        box1, box2 = boxes

    result1 = crop_cxr(image, roi=box1, mask_overlay=mask_overlay, debug_dir=debug_dir)
    result2 = crop_cxr(image, roi=box2, mask_overlay=mask_overlay, debug_dir=debug_dir)
    if result1 is None or result2 is None:
        return None
    return result1, result2


def main():
    """Main entry point for dual CXR cropping."""
    parser = argparse.ArgumentParser(
        description="Crop two chest X-ray regions out of an HDMI capture (side-by-side pane layout)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Crop both CXRs from an image file
  python -m automation.dual_cxr_crop --image captures/two-xray.jpg --out1 /tmp/left.png --out2 /tmp/right.png

  # Crop from a live HDMI capture
  python -m automation.dual_cxr_crop --device 0 --out1 /tmp/left.png --out2 /tmp/right.png

  # Bypass auto-detection
  python -m automation.dual_cxr_crop --image captures/two-xray.jpg \\
      --roi1 132,339,692,665 --roi2 1028,344,692,660 --out1 /tmp/left.png --out2 /tmp/right.png

  # Save intermediate masks to inspect the detection
  python -m automation.dual_cxr_crop --image captures/two-xray.jpg --out1 /tmp/left.png --out2 /tmp/right.png --debug
        """
    )

    # Input options
    parser.add_argument(
        '--image', type=str,
        help='Input image path (default: capture from HDMI)'
    )
    parser.add_argument(
        '--device', type=int,
        help='Video capture device index (default: from config)'
    )

    # Detection options
    parser.add_argument(
        '--roi1', type=_parse_roi,
        help='Manual ROI for the first CXR as x,y,w,h (bypasses auto-detection; requires --roi2)'
    )
    parser.add_argument(
        '--roi2', type=_parse_roi,
        help='Manual ROI for the second CXR as x,y,w,h (bypasses auto-detection; requires --roi1)'
    )
    parser.add_argument(
        '--no-mask-overlay', action='store_true',
        help='Keep the viewer annotations inside the crops'
    )

    # Output options
    parser.add_argument(
        '--out1', type=str,
        help='Path to save the first (left) cropped image'
    )
    parser.add_argument(
        '--out2', type=str,
        help='Path to save the second (right) cropped image'
    )
    parser.add_argument(
        '--debug-dir', type=str,
        help='Directory for --debug images (default: ./automation_outputs/dual_cxr_crop)'
    )

    # Debug options
    parser.add_argument(
        '--debug', action='store_true',
        help='Enable debug logging and save intermediate images'
    )
    parser.add_argument(
        '--env-file', type=str,
        help='Path to .env file'
    )

    args = parser.parse_args()

    if bool(args.roi1) != bool(args.roi2):
        print("--roi1 and --roi2 must be given together", file=sys.stderr)
        return 1

    # Load configuration (password is irrelevant here)
    try:
        config = load_config(args.env_file, skip_password=True)
    except ValueError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        return 1

    if args.device is None:
        args.device = config.capture_device_index

    # Setup logging
    log_level = "DEBUG" if args.debug else config.log_level
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = config.log_dir / f"dual_cxr_crop_{timestamp}.log"
    setup_logging(log_level, log_file, debug_mode=args.debug)

    debug_dir = None
    if args.debug:
        debug_dir = Path(args.debug_dir) if args.debug_dir else (config.output_dir / "dual_cxr_crop")

    # Load the source frame
    if args.image:
        image_path = Path(args.image)
        if not image_path.exists():
            print(f"Image not found: {image_path}", file=sys.stderr)
            return 1
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Failed to read image: {image_path}", file=sys.stderr)
            return 1
        logger.info(f"Loaded image: {image_path} ({image.shape[1]}x{image.shape[0]})")
    else:
        from automation.screen_analyzer import capture_screen

        image = capture_screen(
            device_index=args.device,
            width=config.capture_width,
            height=config.capture_height,
        )
        if image is None:
            print(
                f"HDMI capture failed on device {args.device}. "
                f"Check the connection, or run scripts/warmup_hdmi.py.",
                file=sys.stderr,
            )
            return 1
        logger.info(f"Captured from device {args.device} ({image.shape[1]}x{image.shape[0]})")
        if debug_dir is not None:
            save_debug_image(image, "dual_cxr_source.png", debug_dir)

    results = crop_dual_cxr(
        image,
        roi1=args.roi1,
        roi2=args.roi2,
        mask_overlay=not args.no_mask_overlay,
        debug_dir=debug_dir,
    )

    if results is None:
        print(
            "Could not detect both chest X-ray regions. "
            "Re-run with --roi1 x,y,w,h --roi2 x,y,w,h to specify both regions "
            "manually (--debug saves the intermediate masks).",
            file=sys.stderr,
        )
        return 1

    result1, result2 = results
    print(f"1: {result1.summary()}")
    print(f"2: {result2.summary()}")

    if args.out1:
        out_path = Path(args.out1)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), result1.image)
        print(f"saved: {out_path}")

    if args.out2:
        out_path = Path(args.out2)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), result2.image)
        print(f"saved: {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
