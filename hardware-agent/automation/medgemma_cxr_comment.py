"""
MedGemma CXR Comment

Runs a short, non-diagnostic pulmonary nodule screening comment on a chest
X-ray using google/medgemma-1.5-4b-it, either on an already-cropped CXR
image or on a raw screen capture that is cropped first via
automation.cxr_crop.

Educational / experimental use only. NOT for clinical diagnosis.

Usage:
    # Read an already-cropped CXR image directly
    python -m automation.medgemma_cxr_comment --image captures/cxr.png

    # Crop a screen capture first, then read it
    python -m automation.medgemma_cxr_comment --capture captures/xray.jpg

    # Crop a live HDMI capture first, then read it
    python -m automation.medgemma_cxr_comment --device 0

    # Continuous mode, reusing the loaded model
    python -m automation.medgemma_cxr_comment --watch 10
"""

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from automation.config import load_config
from automation.cxr_crop import CxrCropResult, crop_cxr, _parse_roi
from automation.utils import setup_logging, save_debug_image

logger = logging.getLogger("windows_login")

DEFAULT_MODEL = "google/medgemma-1.5-4b-it"
DEFAULT_MAX_NEW_TOKENS = 300
DEFAULT_PROMPT = (
    "You are reviewing a chest X-ray for an educational, non-diagnostic exercise, "
    "focused specifically on detecting pulmonary nodules. "
    "Carefully examine the lung fields for any pulmonary nodule or mass (a round or "
    "oval opacity within the lung parenchyma). State explicitly whether a pulmonary "
    "nodule is present or absent; if present, briefly note its approximate location "
    "(e.g. right upper lobe). End your answer with a single line starting exactly "
    "with 'Comment:' followed by a 1-2 sentence summary that explicitly states "
    "nodule presence or absence."
)
SYSTEM_PROMPT = "You are an expert radiologist."
DISCLAIMER = "教育・実験用。診断用途ではありません。"


def select_compute_device(preference: str) -> str:
    """Pick mps > cuda > cpu unless a specific device is requested."""
    if preference != "auto":
        return preference
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_medgemma(model_id: str, device: str):
    """Load the MedGemma image-text-to-text pipeline on a single device.

    Uses `device=` (not `device_map="auto"`) so no `accelerate` install is
    required — the 4B model fits entirely on one device.
    """
    import torch
    from transformers import pipeline

    dtype = torch.bfloat16 if device in ("mps", "cuda") else torch.float32
    logger.info(f"Loading {model_id} on {device} ({dtype})...")
    pipe = pipeline("image-text-to-text", model=model_id, dtype=dtype, device=device)
    logger.info("Model loaded")
    return pipe


def run_medgemma(pipe, image: Image.Image, prompt: str, max_new_tokens: int) -> str:
    """Run MedGemma on a single image and return the raw generated text."""
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image", "image": image},
        ]},
    ]
    output = pipe(text=messages, max_new_tokens=max_new_tokens, do_sample=False)
    response = output[0]["generated_text"][-1]["content"]
    if "<unused95>" in response:
        # MedGemma 1.5 thinking trace: <unused94>thought\n ... <unused95> answer
        response = response.split("<unused95>", 1)[1].lstrip()
    return response


def parse_comment(response: str) -> str:
    """Extract the 'Comment:' line, falling back to the full response."""
    for line in response.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("comment:"):
            return stripped.split(":", 1)[1].strip()
    return response.strip()


def bgr_to_pil(image: np.ndarray) -> Image.Image:
    """Convert an OpenCV BGR array to a PIL RGB image."""
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def acquire_cxr(args, config) -> tuple[Optional[np.ndarray], str]:
    """
    Get the CXR image to feed to MedGemma, per the selected input mode.

    Returns:
        (BGR image or None, source description for the output header)
    """
    if args.image:
        image_path = Path(args.image)
        if not image_path.exists():
            print(f"Image not found: {image_path}", file=sys.stderr)
            return None, ""
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Failed to read image: {image_path}", file=sys.stderr)
            return None, ""
        logger.info(f"Loaded CXR image: {image_path} ({image.shape[1]}x{image.shape[0]})")
        return image, f"image={image_path}"

    # --capture / --device: raw screen capture that needs cropping
    if args.capture:
        capture_path = Path(args.capture)
        if not capture_path.exists():
            print(f"Capture image not found: {capture_path}", file=sys.stderr)
            return None, ""
        source = cv2.imread(str(capture_path))
        if source is None:
            print(f"Failed to read capture image: {capture_path}", file=sys.stderr)
            return None, ""
        logger.info(f"Loaded capture: {capture_path} ({source.shape[1]}x{source.shape[0]})")
        source_desc = f"capture={capture_path}"
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
            return None, ""
        logger.info(f"Captured from device {args.device} ({source.shape[1]}x{source.shape[0]})")
        source_desc = f"hdmi device={args.device}"

    debug_dir = args.debug_dir_path
    if debug_dir is not None:
        save_debug_image(source, "medgemma_source.png", debug_dir)

    result: Optional[CxrCropResult] = crop_cxr(
        source,
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
        return None, ""

    x, y, w, h = result.bbox
    logger.info(f"Cropped CXR: method={result.method} bbox={x},{y},{w},{h}")
    return result.image, f"{source_desc} crop={x},{y},{w},{h}"


def report(pipe, image: np.ndarray, source_desc: str, args) -> str:
    """Run inference on one image and print/save the result. Returns the comment text."""
    pil_image = bgr_to_pil(image)
    h, w = image.shape[:2]

    response = run_medgemma(pipe, pil_image, args.prompt, args.max_new_tokens)
    comment = parse_comment(response)

    print("=== MedGemma CXR Comment ===")
    print(f"source: {source_desc} | size={w}x{h}")
    print(f"model: {args.model} ({args.compute_device_resolved})")
    print(f"Comment: {comment}")
    print(f"\n[{DISCLAIMER}]")

    if args.save_dir_path is not None:
        save_debug_image(image, "medgemma_cxr.png", args.save_dir_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        comment_path = args.save_dir_path / f"medgemma_comment_{timestamp}.txt"
        comment_path.parent.mkdir(parents=True, exist_ok=True)
        comment_path.write_text(
            f"source: {source_desc} | size={w}x{h}\n"
            f"model: {args.model} ({args.compute_device_resolved})\n"
            f"Comment: {comment}\n",
            encoding="utf-8",
        )
        logger.info(f"Saved comment: {comment_path}")

    return comment


def main():
    """Main entry point for MedGemma CXR commenting."""
    parser = argparse.ArgumentParser(
        description="Read a chest X-ray with MedGemma and print a short findings comment "
                    "(educational use only, not for diagnosis)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Read an already-cropped CXR image directly
  python -m automation.medgemma_cxr_comment --image captures/cxr.png

  # Crop a screen capture first, then read it
  python -m automation.medgemma_cxr_comment --capture captures/xray.jpg

  # Crop a live HDMI capture first, then read it
  python -m automation.medgemma_cxr_comment --device 0

  # Continuous mode, reusing the loaded model
  python -m automation.medgemma_cxr_comment --watch 10

  # Save capture/crop/comment, with crop debug images
  python -m automation.medgemma_cxr_comment --capture captures/xray.jpg --save-dir ./out --debug
        """
    )

    # Input options
    parser.add_argument(
        '--image', type=str,
        help='Already-cropped CXR image; used as-is (no crop step)'
    )
    parser.add_argument(
        '--capture', type=str,
        help='Raw screen capture image; cropped via automation.cxr_crop first'
    )
    parser.add_argument(
        '--device', type=int,
        help='HDMI capture device index (default: from config). '
             'Used when --image/--capture are omitted'
    )

    # Crop passthrough options (--capture / --device only)
    parser.add_argument(
        '--roi', type=_parse_roi,
        help='Manual crop ROI as x,y,w,h (bypasses auto-detection)'
    )
    parser.add_argument(
        '--full', action='store_true',
        help='Treat the whole capture as the CXR (bypasses auto-detection)'
    )
    parser.add_argument(
        '--no-mask-overlay', action='store_true',
        help='Keep viewer annotations inside the crop'
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
        '--prompt', type=str, default=DEFAULT_PROMPT,
        help='Override the default findings-comment prompt'
    )
    parser.add_argument(
        '--watch', type=float,
        help='Repeat every N seconds, reusing the loaded model (Ctrl+C to stop). '
             'Not compatible with --image'
    )

    # Output options
    parser.add_argument(
        '--save-dir', type=str,
        help='Directory to save capture/crop image and comment text'
    )

    # Debug options
    parser.add_argument(
        '--debug', action='store_true',
        help='Enable debug logging and save intermediate crop images'
    )
    parser.add_argument(
        '--debug-dir', type=str,
        help='Directory for --debug images (default: ./automation_outputs/medgemma_cxr)'
    )
    parser.add_argument(
        '--env-file', type=str,
        help='Path to .env file'
    )

    args = parser.parse_args()

    if args.image and (args.capture or args.device is not None):
        print("--image cannot be combined with --capture/--device", file=sys.stderr)
        return 1
    if args.image and args.watch:
        print("--watch is not compatible with --image (nothing new to capture)", file=sys.stderr)
        return 1
    if args.full and args.roi:
        print("--full and --roi are mutually exclusive", file=sys.stderr)
        return 1

    try:
        config = load_config(args.env_file, skip_password=True)
    except ValueError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        return 1

    if args.device is None:
        args.device = config.capture_device_index

    log_level = "DEBUG" if args.debug else config.log_level
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = config.log_dir / f"medgemma_cxr_comment_{timestamp}.log"
    setup_logging(log_level, log_file, debug_mode=args.debug)

    args.debug_dir_path = None
    if args.debug:
        args.debug_dir_path = Path(args.debug_dir) if args.debug_dir else (config.output_dir / "medgemma_cxr")
    args.save_dir_path = Path(args.save_dir) if args.save_dir else None

    args.compute_device_resolved = select_compute_device(args.compute_device)

    if not args.watch:
        # Fail fast on a bad crop before paying the (multi-GB) model load cost.
        image, source_desc = acquire_cxr(args, config)
        if image is None:
            return 1

        try:
            pipe = load_medgemma(args.model, args.compute_device_resolved)
        except Exception as e:
            print(f"Failed to load {args.model}: {e}", file=sys.stderr)
            return 1

        report(pipe, image, source_desc, args)
        return 0

    # --watch: load once, keep retrying acquisition each cycle (e.g. while
    # waiting for the HDMI source to show a valid CXR).
    try:
        pipe = load_medgemma(args.model, args.compute_device_resolved)
    except Exception as e:
        print(f"Failed to load {args.model}: {e}", file=sys.stderr)
        return 1

    logger.info(f"Watch mode: every {args.watch}s (Ctrl+C to stop)")
    try:
        while True:
            image, source_desc = acquire_cxr(args, config)
            if image is not None:
                report(pipe, image, source_desc, args)
            else:
                logger.warning("Skipping this cycle (no CXR acquired)")
            time.sleep(args.watch)
    except KeyboardInterrupt:
        logger.info("Stopped")
        return 0


if __name__ == "__main__":
    sys.exit(main())
