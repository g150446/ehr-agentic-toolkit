"""
MedGemma CXR Longitudinal Comparison

Compares two chest X-rays of the same patient (a newer image vs. an older
one) using google/medgemma-1.5-4b-it, and asks whether a new pulmonary
nodule has appeared in the newer image. Images can be given directly as
already-cropped CXR files, auto-cropped from an existing side-by-side
dual-pane screen image, or captured live from HDMI.

Educational / experimental use only. NOT for clinical diagnosis.

Usage:
    # Recommended: capture the current dual-pane HDMI view from device 0
    # (device 0 is the default, so --device is not required)
    python -m automation.medgemma_cxr_compare --capture

    # Capture from a different HDMI device
    python -m automation.medgemma_cxr_compare --capture --device 1

    # Auto-crop both panes from an existing dual-pane screen image
    python -m automation.medgemma_cxr_compare --image captures/two-xray.jpg

    # Two already-cropped CXR images (left/new vs right/old)
    python -m automation.medgemma_cxr_compare --new-image /tmp/two_left.png --old-image /tmp/two_right.png
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import cv2
from PIL import Image

from automation.config import load_config
from automation.cxr_crop import _parse_roi
from automation.dual_cxr_crop import crop_dual_cxr
from automation.medgemma_cxr_comment import (
    DEFAULT_MODEL,
    DEFAULT_MAX_NEW_TOKENS,
    SYSTEM_PROMPT,
    DISCLAIMER,
    select_compute_device,
    load_medgemma,
    bgr_to_pil,
    parse_comment,
)
from automation.utils import setup_logging, save_debug_image

logger = logging.getLogger("windows_login")

DEFAULT_COMPARE_PROMPT = (
    "You are comparing two chest X-rays of the same patient taken at different "
    "times, for an educational, non-diagnostic exercise focused specifically on "
    "detecting a newly appeared pulmonary nodule. The first image is the newer "
    "(current) X-ray; the second image is the older (prior) X-ray for comparison. "
    "Carefully compare the lung fields between the two images and identify any "
    "pulmonary nodule or mass (a round or oval opacity within the lung "
    "parenchyma) that is visible in the new image but was NOT present in the "
    "old image. State explicitly whether a new pulmonary nodule has appeared; "
    "if so, briefly note its approximate location (e.g. right upper lobe). End "
    "your answer with a single line starting exactly with 'Comment:' followed "
    "by a 1-2 sentence summary that explicitly states whether a new nodule has "
    "appeared."
)


def run_medgemma_compare(
    pipe,
    new_image: Image.Image,
    old_image: Image.Image,
    prompt: str,
    max_new_tokens: int,
) -> str:
    """Run MedGemma on two images (new, old) and return the raw generated text."""
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [
            {"type": "image", "image": new_image},
            {"type": "image", "image": old_image},
            {"type": "text", "text": prompt},
        ]},
    ]
    output = pipe(text=messages, max_new_tokens=max_new_tokens, do_sample=False)
    response = output[0]["generated_text"][-1]["content"]
    if "<unused95>" in response:
        # MedGemma 1.5 thinking trace: <unused94>thought\n ... <unused95> answer
        response = response.split("<unused95>", 1)[1].lstrip()
    return response


def acquire_cxr_pair(args, config) -> Tuple[Optional["cv2.Mat"], Optional["cv2.Mat"], str]:
    """
    Get the (new, old) CXR image pair to feed to MedGemma, per the selected input mode.

    Returns:
        (new BGR image or None, old BGR image or None, source description)
    """
    if args.new_image:
        new_path = Path(args.new_image)
        old_path = Path(args.old_image)
        if not new_path.exists():
            print(f"Image not found: {new_path}", file=sys.stderr)
            return None, None, ""
        if not old_path.exists():
            print(f"Image not found: {old_path}", file=sys.stderr)
            return None, None, ""
        new_image = cv2.imread(str(new_path))
        old_image = cv2.imread(str(old_path))
        if new_image is None:
            print(f"Failed to read image: {new_path}", file=sys.stderr)
            return None, None, ""
        if old_image is None:
            print(f"Failed to read image: {old_path}", file=sys.stderr)
            return None, None, ""
        logger.info(f"Loaded new={new_path} ({new_image.shape[1]}x{new_image.shape[0]}) "
                    f"old={old_path} ({old_image.shape[1]}x{old_image.shape[0]})")
        return new_image, old_image, f"new={new_path} old={old_path}"

    # --image / --capture: raw dual-pane screen image that needs cropping
    if args.image:
        image_path = Path(args.image)
        if not image_path.exists():
            print(f"Screen image not found: {image_path}", file=sys.stderr)
            return None, None, ""
        source = cv2.imread(str(image_path))
        if source is None:
            print(f"Failed to read screen image: {image_path}", file=sys.stderr)
            return None, None, ""
        logger.info(f"Loaded screen image: {image_path} ({source.shape[1]}x{source.shape[0]})")
        source_desc = f"image={image_path}"
    else:
        from automation.screen_analyzer import capture_screen

        source = capture_screen(
            device_index=args.device,
            width=config.capture_width,
            height=config.capture_height,
        )
        if source is None:
            print(
                f"HDMI capture failed on device {args.device}. "
                f"Check the connection, or run scripts/warmup_hdmi.py.",
                file=sys.stderr,
            )
            return None, None, ""
        logger.info(f"Captured from device {args.device} ({source.shape[1]}x{source.shape[0]})")
        source_desc = f"hdmi device={args.device}"

    debug_dir = args.debug_dir_path
    if debug_dir is not None:
        save_debug_image(source, "medgemma_compare_source.png", debug_dir)

    results = crop_dual_cxr(
        source,
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
        return None, None, ""

    new_result, old_result = results
    nx, ny, nw, nh = new_result.bbox
    ox, oy, ow, oh = old_result.bbox
    logger.info(f"Cropped new: method={new_result.method} bbox={nx},{ny},{nw},{nh}")
    logger.info(f"Cropped old: method={old_result.method} bbox={ox},{oy},{ow},{oh}")
    return new_result.image, old_result.image, f"{source_desc} new={nx},{ny},{nw},{nh} old={ox},{oy},{ow},{oh}"


def report(pipe, new_image, old_image, source_desc: str, args) -> str:
    """Run comparison inference and print/save the result. Returns the comment text."""
    new_pil = bgr_to_pil(new_image)
    old_pil = bgr_to_pil(old_image)
    nh, nw = new_image.shape[:2]
    oh, ow = old_image.shape[:2]

    response = run_medgemma_compare(pipe, new_pil, old_pil, args.prompt, args.max_new_tokens)
    comment = parse_comment(response)

    print("=== MedGemma CXR Comparison ===")
    print(f"source: {source_desc}")
    print(f"new size={nw}x{nh} | old size={ow}x{oh}")
    print(f"model: {args.model} ({args.compute_device_resolved})")
    print(f"Comment: {comment}")
    print(f"\n[{DISCLAIMER}]")

    if args.save_dir_path is not None:
        save_debug_image(new_image, "medgemma_compare_new.png", args.save_dir_path)
        save_debug_image(old_image, "medgemma_compare_old.png", args.save_dir_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        comment_path = args.save_dir_path / f"medgemma_compare_comment_{timestamp}.txt"
        comment_path.parent.mkdir(parents=True, exist_ok=True)
        comment_path.write_text(
            f"source: {source_desc}\n"
            f"new size={nw}x{nh} | old size={ow}x{oh}\n"
            f"model: {args.model} ({args.compute_device_resolved})\n"
            f"Comment: {comment}\n",
            encoding="utf-8",
        )
        logger.info(f"Saved comment: {comment_path}")

    return comment


def main():
    """Main entry point for MedGemma CXR longitudinal comparison."""
    parser = argparse.ArgumentParser(
        description="Compare two chest X-rays with MedGemma and report whether a new "
                    "pulmonary nodule has appeared (educational use only, not for diagnosis)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Two already-cropped CXR images (left/new vs right/old)
  python -m automation.medgemma_cxr_compare --new-image /tmp/two_left.png --old-image /tmp/two_right.png

  # Auto-crop both panes from an existing dual-pane screen image
  python -m automation.medgemma_cxr_compare --image captures/two-xray.jpg

  # Capture the dual-pane view from live HDMI, then auto-crop it
  python -m automation.medgemma_cxr_compare --capture --device 0

  # Save both crops + comparison text, with crop debug images
  python -m automation.medgemma_cxr_compare --image captures/two-xray.jpg --save-dir ./out --debug
        """
    )

    # Input options
    parser.add_argument(
        '--new-image', type=str,
        help='Already-cropped newer CXR image; used as-is (no crop step)'
    )
    parser.add_argument(
        '--old-image', type=str,
        help='Already-cropped older CXR image; used as-is (no crop step)'
    )
    parser.add_argument(
        '--image', type=str,
        help='Existing dual-pane screen image; cropped via automation.dual_cxr_crop first'
    )
    parser.add_argument(
        '--capture', action='store_true',
        help='Capture one live HDMI frame and crop both CXR panes from it'
    )
    parser.add_argument(
        '--device', type=int,
        help='HDMI capture device index for --capture (default: 0)'
    )

    # Crop passthrough options (--image / --capture only)
    parser.add_argument(
        '--roi1', type=_parse_roi,
        help='Manual crop ROI for the new (left) CXR as x,y,w,h (bypasses auto-detection; requires --roi2)'
    )
    parser.add_argument(
        '--roi2', type=_parse_roi,
        help='Manual crop ROI for the old (right) CXR as x,y,w,h (bypasses auto-detection; requires --roi1)'
    )
    parser.add_argument(
        '--no-mask-overlay', action='store_true',
        help='Keep viewer annotations inside the crops'
    )

    # MedGemma options
    parser.add_argument(
        '--model', type=str, default=DEFAULT_MODEL,
        help=f'Hugging Face model ID (default: {DEFAULT_MODEL})'
    )
    parser.add_argument(
        '--compute-device', choices=['auto', 'mps', 'cuda', 'cpu'], default='auto',
        help='Inference device (default: auto = mps > cuda > cpu)'
    )
    parser.add_argument(
        '--max-new-tokens', type=int, default=DEFAULT_MAX_NEW_TOKENS,
        help=f'Max generated tokens (default: {DEFAULT_MAX_NEW_TOKENS})'
    )
    parser.add_argument(
        '--prompt', type=str, default=DEFAULT_COMPARE_PROMPT,
        help='Override the default comparison prompt'
    )

    # Output options
    parser.add_argument(
        '--save-dir', type=str,
        help='Directory to save both crop images and the comparison text'
    )

    # Debug options
    parser.add_argument(
        '--debug', action='store_true',
        help='Enable debug logging and save intermediate crop images'
    )
    parser.add_argument(
        '--debug-dir', type=str,
        help='Directory for --debug images (default: ./automation_outputs/medgemma_cxr_compare)'
    )
    parser.add_argument(
        '--env-file', type=str,
        help='Path to .env file'
    )

    args = parser.parse_args()

    if bool(args.new_image) != bool(args.old_image):
        print("--new-image and --old-image must be given together", file=sys.stderr)
        return 1
    direct_mode = bool(args.new_image)
    if args.device is not None and not args.capture:
        print("--device requires --capture", file=sys.stderr)
        return 1
    input_mode_count = sum((direct_mode, bool(args.image), args.capture))
    if input_mode_count == 0:
        print(
            "Specify one input mode: --new-image/--old-image, --image, or --capture",
            file=sys.stderr,
        )
        return 1
    if input_mode_count > 1:
        print(
            "--new-image/--old-image, --image, and --capture are mutually exclusive",
            file=sys.stderr,
        )
        return 1
    if direct_mode and (args.roi1 or args.roi2):
        print("--roi1/--roi2 only apply to --image/--capture mode", file=sys.stderr)
        return 1
    if bool(args.roi1) != bool(args.roi2):
        print("--roi1 and --roi2 must be given together", file=sys.stderr)
        return 1

    try:
        config = load_config(args.env_file, skip_password=True)
    except ValueError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        return 1

    if args.capture and args.device is None:
        args.device = 0

    log_level = "DEBUG" if args.debug else config.log_level
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = config.log_dir / f"medgemma_cxr_compare_{timestamp}.log"
    setup_logging(log_level, log_file, debug_mode=args.debug)

    args.debug_dir_path = None
    if args.debug:
        args.debug_dir_path = Path(args.debug_dir) if args.debug_dir else (config.output_dir / "medgemma_cxr_compare")
    args.save_dir_path = Path(args.save_dir) if args.save_dir else None

    args.compute_device_resolved = select_compute_device(args.compute_device)

    # Fail fast on a bad crop/read before paying the (multi-GB) model load cost.
    new_image, old_image, source_desc = acquire_cxr_pair(args, config)
    if new_image is None or old_image is None:
        return 1

    try:
        pipe = load_medgemma(args.model, args.compute_device_resolved)
    except Exception as e:
        print(f"Failed to load {args.model}: {e}", file=sys.stderr)
        return 1

    report(pipe, new_image, old_image, source_desc, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
