"""
CXR Crop

Crops the chest X-ray (CXR) region out of an HDMI capture of the PSP Viewer
reading screen (読影画面), and blacks out the viewer's burned-in annotations.

Detection is layout-driven rather than model-driven:

  1. Find the blue selection frame the viewer draws around the active image pane.
  2. Inside that pane, find the large achromatic bright blob (= the radiograph).
  3. Reject candidates that are too small / too large / wrongly shaped.
  4. Keep only the pixels connected to the anatomy, which blacks out every
     overlay annotation drawn on the radiograph's black background.

Usage:
    # Crop from an image file
    python -m automation.cxr_crop --image captures/xray.jpg --out /tmp/cxr.png

    # Crop from a live HDMI capture
    python -m automation.cxr_crop --device 0 --out /tmp/cxr.png

    # Bypass auto-detection
    python -m automation.cxr_crop --image captures/xray.jpg --roi 600,330,692,674 --out /tmp/cxr.png
    python -m automation.cxr_crop --image captures/xray.jpg --full --out /tmp/cxr.png
"""

import argparse
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from automation.config import load_config
from automation.utils import setup_logging, save_debug_image

logger = logging.getLogger("windows_login")

# --- Step 1: blue selection frame ---------------------------------------
# The viewer draws a pure blue frame (BGR ~= [220, 2, 2]) around the active
# pane. A looser mask picks up the title bar gradient, so keep G/R low.
BLUE_FRAME_B_MIN = 100
BLUE_FRAME_GR_MAX = 90
BLUE_FRAME_MIN_W = 400
BLUE_FRAME_MIN_H = 300
FRAME_INSET = 12  # frame thickness; inset this far to get the canvas

# --- Step 2: achromatic bright blob -------------------------------------
# The saturation condition is what separates the radiograph from the UI
# (CXR: mean S ~= 1, UI: mean S ~= 40 with p95 = 255). Without it the whole
# toolbar/list/taskbar merges into one component that swallows the screen.
BRIGHT_MIN = 25
SAT_MAX = 40
OPEN_KERNEL = 15  # erases overlay text, ruler ticks and thin UI rules
CLOSE_KERNEL = 9  # smooths the anatomy outline before masking

# --- Step 3: candidate filter -------------------------------------------
MIN_SIDE = 300  # excludes the ~75x115 thumbnail in the right-hand strip
MAX_W_RATIO = 0.9
MAX_H_RATIO = 0.95
ASPECT_MIN = 0.5
ASPECT_MAX = 2.0

# --- Step 5: sanity checks ----------------------------------------------
MAX_SAT_MEAN = 10  # not a grayscale image
MIN_GRAY_STD = 20  # flat, featureless region

Box = Tuple[int, int, int, int]  # (x, y, w, h)


@dataclass
class CxrCropResult:
    """Result of a CXR crop attempt."""

    image: np.ndarray
    bbox: Box
    viewport: Optional[Box]
    method: str  # blue-frame | fullscreen-fallback | manual-roi | full
    masked_pixels: int

    def summary(self) -> str:
        """One-line description for stdout."""
        x, y, w, h = self.bbox
        vp = ",".join(str(v) for v in self.viewport) if self.viewport else "none"
        return (
            f"cxr: method={self.method} viewport={vp} "
            f"bbox={x},{y},{w},{h} size={w}x{h} masked={self.masked_pixels}px"
        )


def _bright_achromatic_mask(image: np.ndarray) -> np.ndarray:
    """Binary mask of bright, low-saturation (grayscale) pixels."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sat = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:, :, 1]
    return ((gray > BRIGHT_MIN) & (sat < SAT_MAX)).astype(np.uint8) * 255


def find_viewer_viewport(image: np.ndarray) -> Optional[Box]:
    """
    Find the blue selection frame around the viewer's active image pane.

    Args:
        image: Full screen capture (BGR)

    Returns:
        (x, y, w, h) of the frame itself, or None if no frame is present
    """
    b = image[:, :, 0].astype(np.int16)
    g = image[:, :, 1].astype(np.int16)
    r = image[:, :, 2].astype(np.int16)
    blue = (
        (b > BLUE_FRAME_B_MIN)
        & (g < BLUE_FRAME_GR_MAX)
        & (r < BLUE_FRAME_GR_MAX)
    ).astype(np.uint8) * 255

    # Dilate so the four edges of the frame form a single component
    blue = cv2.dilate(blue, np.ones((3, 3), np.uint8))
    count, _, stats, _ = cv2.connectedComponentsWithStats(blue, 8)

    best: Optional[Box] = None
    best_area = 0
    for i in range(1, count):
        x, y, w, h, area = stats[i]
        if w < BLUE_FRAME_MIN_W or h < BLUE_FRAME_MIN_H:
            continue
        if area > best_area:
            best_area, best = area, (int(x), int(y), int(w), int(h))

    if best is None:
        logger.debug("No blue viewer frame found")
    else:
        logger.debug(f"Blue viewer frame: {best}")
    return best


def _passes_sanity_check(image: np.ndarray, box: Box) -> bool:
    """Reject regions that cannot plausibly be a radiograph."""
    x, y, w, h = box
    region = image[y:y + h, x:x + w]
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    sat_mean = float(cv2.cvtColor(region, cv2.COLOR_BGR2HSV)[:, :, 1].mean())
    gray_std = float(gray.std())

    if sat_mean > MAX_SAT_MEAN:
        logger.debug(f"Rejected {box}: mean saturation {sat_mean:.1f} > {MAX_SAT_MEAN}")
        return False
    if gray_std < MIN_GRAY_STD:
        logger.debug(f"Rejected {box}: gray std {gray_std:.1f} < {MIN_GRAY_STD}")
        return False
    return True


def find_cxr_region(
    image: np.ndarray,
    viewport: Optional[Box] = None,
    debug_dir: Optional[Path] = None,
) -> Optional[Box]:
    """
    Locate the radiograph inside a screen capture.

    Args:
        image: Full screen capture (BGR)
        viewport: Blue frame from find_viewer_viewport(). If given, the search
                  is restricted to the canvas inside it.
        debug_dir: If set, the intermediate blob mask is saved here

    Returns:
        (x, y, w, h) in full-screen coordinates, or None if nothing qualifies
    """
    full_h, full_w = image.shape[:2]

    if viewport is not None:
        vx, vy, vw, vh = viewport
        ox, oy = vx + FRAME_INSET, vy + FRAME_INSET
        canvas = image[oy:vy + vh - FRAME_INSET, ox:vx + vw - FRAME_INSET]
    else:
        ox, oy = 0, 0
        canvas = image

    if canvas.size == 0:
        logger.warning("Viewport canvas is empty")
        return None

    mask = _bright_achromatic_mask(canvas)
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, np.ones((OPEN_KERNEL, OPEN_KERNEL), np.uint8)
    )
    if debug_dir is not None:
        save_debug_image(mask, "cxr_blob_mask.png", debug_dir)

    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)

    best: Optional[Box] = None
    best_area = 0
    for i in range(1, count):
        x, y, w, h, area = stats[i]
        if w < MIN_SIDE or h < MIN_SIDE:
            continue
        if w > MAX_W_RATIO * full_w or h > MAX_H_RATIO * full_h:
            continue
        aspect = w / h
        if not (ASPECT_MIN < aspect < ASPECT_MAX):
            continue
        if area <= best_area:
            continue
        best_area, best = area, (int(x) + ox, int(y) + oy, int(w), int(h))

    if best is None:
        logger.debug("No CXR candidate passed the size/aspect filter")
        return None
    if not _passes_sanity_check(image, best):
        return None

    logger.debug(f"CXR region: {best} (area={best_area})")
    return best


def mask_overlay_annotations(crop: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Black out everything not connected to the anatomy.

    The viewer's burned-in annotations (e.g. "立位 P→A") sit on the
    radiograph's black background, so they are never part of the anatomy's
    connected component and are removed without touching the anatomy itself.

    Annotations drawn *on top of* the lung fields stay connected and survive;
    use --roi to work around those.

    Args:
        crop: Cropped CXR region (BGR)

    Returns:
        (masked copy, number of bright pixels removed)
    """
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    raw = (gray > BRIGHT_MIN).astype(np.uint8)

    # Same opening as detection: only the anatomy blob survives it
    seed = cv2.morphologyEx(
        raw * 255, cv2.MORPH_OPEN, np.ones((OPEN_KERNEL, OPEN_KERNEL), np.uint8)
    ) > 0
    if not seed.any():
        logger.debug("Overlay masking skipped: no anatomy seed")
        return crop.copy(), 0

    count, labels, _, _ = cv2.connectedComponentsWithStats(raw, 8)
    best_overlap, best_label = 0, 0
    for i in range(1, count):
        overlap = int(((labels == i) & seed).sum())
        if overlap > best_overlap:
            best_overlap, best_label = overlap, i

    if best_label == 0:
        logger.debug("Overlay masking skipped: no component overlaps the seed")
        return crop.copy(), 0

    keep = (labels == best_label).astype(np.uint8)
    keep = cv2.morphologyEx(
        keep, cv2.MORPH_CLOSE, np.ones((CLOSE_KERNEL, CLOSE_KERNEL), np.uint8)
    )

    removed = int(((gray > 200) & (keep == 0)).sum())
    masked = crop * keep[:, :, None]
    logger.debug(f"Overlay masking removed {removed} bright pixels")
    return masked, removed


def crop_cxr(
    image: np.ndarray,
    roi: Optional[Box] = None,
    full: bool = False,
    mask_overlay: bool = True,
    debug_dir: Optional[Path] = None,
) -> Optional[CxrCropResult]:
    """
    Crop the chest X-ray region out of a screen capture.

    Args:
        image: Full screen capture (BGR)
        roi: Manual (x, y, w, h). Bypasses auto-detection.
        full: Treat the whole image as the CXR. Bypasses auto-detection.
        mask_overlay: Black out viewer annotations inside the crop
        debug_dir: If set, intermediate images are saved here

    Returns:
        CxrCropResult, or None if auto-detection found no CXR
    """
    full_h, full_w = image.shape[:2]
    viewport: Optional[Box] = None

    if full:
        bbox, method = (0, 0, full_w, full_h), "full"
    elif roi is not None:
        x, y, w, h = roi
        x, y = max(0, x), max(0, y)
        w, h = min(w, full_w - x), min(h, full_h - y)
        if w <= 0 or h <= 0:
            logger.error(f"ROI {roi} lies outside the {full_w}x{full_h} image")
            return None
        bbox, method = (x, y, w, h), "manual-roi"
    else:
        viewport = find_viewer_viewport(image)
        bbox = find_cxr_region(image, viewport, debug_dir)
        method = "blue-frame"

        if bbox is None and viewport is not None:
            # The frame may be there but the pane empty; retry on the full screen
            logger.debug("Retrying CXR detection without the viewport constraint")
            viewport = None
            bbox = find_cxr_region(image, None, debug_dir)
            method = "fullscreen-fallback"
        elif viewport is None:
            method = "fullscreen-fallback"

        if bbox is None:
            return None

    x, y, w, h = bbox
    crop = image[y:y + h, x:x + w].copy()

    removed = 0
    if mask_overlay:
        crop, removed = mask_overlay_annotations(crop)

    if debug_dir is not None:
        overlay = image.copy()
        if viewport is not None:
            vx, vy, vw, vh = viewport
            cv2.rectangle(overlay, (vx, vy), (vx + vw, vy + vh), (0, 255, 255), 2)
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 3)
        save_debug_image(overlay, "cxr_detection.png", debug_dir)

    return CxrCropResult(
        image=crop,
        bbox=(x, y, w, h),
        viewport=viewport,
        method=method,
        masked_pixels=removed,
    )


def _parse_roi(value: str) -> Box:
    """Parse an 'x,y,w,h' CLI argument."""
    parts = value.split(",")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(f"ROI must be 'x,y,w,h', got: {value}")
    try:
        x, y, w, h = (int(p) for p in parts)
    except ValueError:
        raise argparse.ArgumentTypeError(f"ROI values must be integers: {value}")
    if w <= 0 or h <= 0:
        raise argparse.ArgumentTypeError(f"ROI width/height must be positive: {value}")
    return (x, y, w, h)


def main():
    """Main entry point for CXR cropping."""
    parser = argparse.ArgumentParser(
        description="Crop the chest X-ray region out of an HDMI capture",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Crop from an image file
  python -m automation.cxr_crop --image captures/xray.jpg --out /tmp/cxr.png

  # Crop from a live HDMI capture
  python -m automation.cxr_crop --device 0 --out /tmp/cxr.png

  # Bypass auto-detection
  python -m automation.cxr_crop --image captures/xray.jpg --roi 600,330,692,674 --out /tmp/cxr.png
  python -m automation.cxr_crop --image captures/xray.jpg --full --out /tmp/cxr.png

  # Save intermediate masks to inspect the detection
  python -m automation.cxr_crop --image captures/xray.jpg --out /tmp/cxr.png --debug
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
        '--roi', type=_parse_roi,
        help='Manual ROI as x,y,w,h (bypasses auto-detection)'
    )
    parser.add_argument(
        '--full', action='store_true',
        help='Treat the whole image as the CXR (bypasses auto-detection)'
    )
    parser.add_argument(
        '--no-mask-overlay', action='store_true',
        help='Keep the viewer annotations inside the crop'
    )

    # Output options
    parser.add_argument(
        '--out', type=str,
        help='Path to save the cropped image'
    )
    parser.add_argument(
        '--debug-dir', type=str,
        help='Directory for --debug images (default: ./automation_outputs/cxr_crop)'
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

    if args.full and args.roi:
        print("--full and --roi are mutually exclusive", file=sys.stderr)
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
    log_file = config.log_dir / f"cxr_crop_{timestamp}.log"
    setup_logging(log_level, log_file, debug_mode=args.debug)

    debug_dir = None
    if args.debug:
        debug_dir = Path(args.debug_dir) if args.debug_dir else (config.output_dir / "cxr_crop")

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
            save_debug_image(image, "cxr_source.png", debug_dir)

    result = crop_cxr(
        image,
        roi=args.roi,
        full=args.full,
        mask_overlay=not args.no_mask_overlay,
        debug_dir=debug_dir,
    )

    if result is None:
        print(
            "No chest X-ray region detected. "
            "Re-run with --full to use the whole screen, or --roi x,y,w,h to "
            "specify the region manually (--debug saves the intermediate masks).",
            file=sys.stderr,
        )
        return 1

    print(result.summary())

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), result.image)
        print(f"saved: {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
