#!/usr/bin/env python3
"""
HDMI Capture Stream Monitor

Real-time video streaming tool for HDMI capture devices with optional YOLO detection overlay.

Usage:
    python -m automation.monitor_stream [--detection-on] [--fps 5] [--debug]
    ./run_monitor.sh [--detection-on] [--fps 5]

Keyboard Controls:
    Q/ESC - Quit application
    D     - Toggle YOLO detection
    S     - Save screenshot
    F     - Toggle FPS counter
    H     - Toggle help overlay
    +     - Increase confidence threshold
    -     - Decrease confidence threshold
"""

import argparse
import sys
import logging
import time
import cv2
import numpy as np
from pathlib import Path
from collections import deque
from datetime import datetime
from typing import Optional, List, Tuple

# Import automation modules
from automation.config import load_config, AutomationConfig
from automation.screen_analyzer import (
    load_yolo_model,
    analyze_layout,
    visualize_detections,
    DetectedRegion
)
from automation.utils import (
    setup_logging,
    save_debug_image
)


logger = logging.getLogger("monitor_stream")


class StreamMonitor:
    """Real-time HDMI capture monitor with optional YOLO detection."""

    def __init__(self, config: AutomationConfig, args: argparse.Namespace):
        """
        Initialize stream monitor.

        Args:
            config: Automation configuration
            args: Command-line arguments
        """
        self.config = config
        self.args = args

        # State
        self.running = False
        self.detection_enabled = args.detection_on
        self.show_fps = True
        self.show_help = not args.no_help
        self.confidence = args.confidence

        # Resources (lazy-loaded)
        self.capture = None
        self.yolo_model = None

        # FPS tracking (rolling 30-frame average)
        self.frame_times = deque(maxlen=30)

        # Statistics
        self.total_frames = 0
        self.total_detections = 0
        self.consecutive_failures = 0

        # Create output directory
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def initialize_capture(self) -> cv2.VideoCapture:
        """
        Initialize video capture device.

        Returns:
            Opened VideoCapture object

        Raises:
            RuntimeError: If device cannot be opened
        """
        logger.info(f"Opening capture device {self.args.device}...")

        cap = cv2.VideoCapture(self.args.device)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open device {self.args.device}")

        # Configure resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.capture_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.capture_height)

        # Get actual resolution (may differ from requested)
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        backend = cap.getBackendName()

        logger.info(f"Capture initialized: {actual_width}x{actual_height} ({backend})")

        # Warmup frames (device stabilization)
        logger.info("Warming up capture device...")
        for i in range(5):
            cap.read()
            time.sleep(0.1)

        return cap

    def ensure_yolo_loaded(self):
        """
        Lazy-load YOLO model when first needed.

        Returns:
            Loaded YOLO model
        """
        if self.yolo_model is None:
            logger.info("Loading DocLayout-YOLO model...")
            try:
                self.yolo_model = load_yolo_model(
                    self.config.doclayout_model_path,
                    self.config.detection_device
                )
                logger.info("YOLO model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load YOLO model: {e}")
                logger.warning("Detection mode disabled due to model loading failure")
                self.detection_enabled = False
                return None
        return self.yolo_model

    def process_frame(
        self,
        frame: np.ndarray
    ) -> Tuple[np.ndarray, List[DetectedRegion]]:
        """
        Process frame with optional YOLO detection.

        Args:
            frame: Input frame (BGR format)

        Returns:
            Tuple of (processed_frame, detected_regions)
        """
        regions = []

        if self.detection_enabled:
            model = self.ensure_yolo_loaded()
            if model is not None:
                try:
                    regions = analyze_layout(
                        frame,
                        model,
                        confidence=self.confidence,
                        image_size=self.args.imgsz
                    )
                    frame = visualize_detections(frame, regions)
                    self.total_detections += len(regions)
                except Exception as e:
                    logger.error(f"Detection error: {e}")

        return frame, regions

    def draw_text_with_background(
        self,
        img: np.ndarray,
        text: str,
        pos: Tuple[int, int],
        fg_color: Tuple[int, int, int] = (255, 255, 255),
        bg_color: Tuple[int, int, int] = (0, 0, 0),
        font_scale: float = 0.6,
        thickness: int = 2
    ) -> np.ndarray:
        """
        Draw text with background rectangle for better visibility.

        Args:
            img: Image to draw on
            text: Text to draw
            pos: Position (x, y)
            fg_color: Foreground color (BGR)
            bg_color: Background color (BGR)
            font_scale: Font scale
            thickness: Text thickness

        Returns:
            Modified image
        """
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Calculate text size
        (text_width, text_height), _ = cv2.getTextSize(
            text, font, font_scale, thickness
        )

        # Draw background rectangle
        x, y = pos
        cv2.rectangle(
            img,
            (x - 5, y - text_height - 5),
            (x + text_width + 5, y + 5),
            bg_color,
            -1
        )

        # Draw text
        cv2.putText(img, text, pos, font, font_scale, fg_color, thickness, cv2.LINE_AA)
        return img

    def draw_overlay(
        self,
        frame: np.ndarray,
        fps: float,
        num_detections: int
    ) -> np.ndarray:
        """
        Draw status overlay on frame.

        Args:
            frame: Input frame
            fps: Current FPS
            num_detections: Number of detected regions

        Returns:
            Frame with overlay
        """
        result = frame.copy()
        h, w = frame.shape[:2]

        # Status bar (top-left)
        if self.show_fps:
            mode = "DETECTION" if self.detection_enabled else "RAW"
            status_lines = [
                f"Mode: {mode}",
                f"FPS: {fps:.1f}",
                f"Device: {self.args.device} ({w}x{h})",
            ]

            if self.detection_enabled:
                status_lines.append(f"Detected: {num_detections} regions")
                status_lines.append(f"Conf: {self.confidence:.2f}")

            y_offset = 30
            for line in status_lines:
                result = self.draw_text_with_background(
                    result, line, (10, y_offset), (0, 255, 0)
                )
                y_offset += 30

        # Help overlay (bottom-right)
        if self.show_help:
            help_lines = [
                "Controls:",
                "  Q/ESC - Quit",
                "  D - Detection",
                "  S - Screenshot",
                "  F - FPS",
                "  H - Help",
                "  +/- - Confidence",
            ]

            # Calculate starting y position
            y_offset = h - 30 * len(help_lines) - 10

            for line in help_lines:
                result = self.draw_text_with_background(
                    result, line, (w - 250, y_offset), (255, 255, 0)
                )
                y_offset += 30

        return result

    def handle_keyboard(self, key: int) -> str:
        """
        Handle keyboard input.

        Args:
            key: Key code from cv2.waitKey

        Returns:
            Action name ('quit', 'toggle_detect', 'screenshot', etc.)
        """
        if key == ord('q') or key == 27:  # Q or ESC
            return 'quit'
        elif key == ord('d'):
            self.detection_enabled = not self.detection_enabled
            logger.info(f"Detection: {'ON' if self.detection_enabled else 'OFF'}")
            return 'toggle_detect'
        elif key == ord('s'):
            return 'screenshot'
        elif key == ord('f'):
            self.show_fps = not self.show_fps
            logger.info(f"FPS display: {'ON' if self.show_fps else 'OFF'}")
            return 'toggle_fps'
        elif key == ord('h'):
            self.show_help = not self.show_help
            logger.info(f"Help overlay: {'ON' if self.show_help else 'OFF'}")
            return 'toggle_help'
        elif key == ord('+') or key == ord('='):
            self.confidence = min(1.0, self.confidence + 0.05)
            logger.info(f"Confidence: {self.confidence:.2f}")
            return 'increase_conf'
        elif key == ord('-') or key == ord('_'):
            self.confidence = max(0.0, self.confidence - 0.05)
            logger.info(f"Confidence: {self.confidence:.2f}")
            return 'decrease_conf'
        return 'none'

    def save_screenshot(self, frame: np.ndarray) -> None:
        """
        Save current frame as screenshot.

        Args:
            frame: Frame to save
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"monitor_screenshot_{timestamp}.jpg"
        path = self.output_dir / filename

        cv2.imwrite(str(path), frame)
        logger.info(f"Screenshot saved: {path}")

    def run(self) -> None:
        """Main monitoring loop."""
        try:
            # Initialize
            self.capture = self.initialize_capture()
            window_name = "HDMI Capture Monitor (Press H for help)"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

            logger.info("=" * 70)
            logger.info("Monitor started. Press 'q' to quit.")
            logger.info(f"Target FPS: {self.args.fps}")
            logger.info(f"Detection: {'ON' if self.detection_enabled else 'OFF'}")
            logger.info("=" * 70)

            self.running = True
            target_interval = 1.0 / self.args.fps

            while self.running:
                loop_start = time.time()

                # Capture frame
                ret, frame = self.capture.read()

                if not ret or frame is None:
                    self.consecutive_failures += 1
                    logger.warning(f"Frame capture failed (attempt {self.consecutive_failures})")

                    if self.consecutive_failures > 10:
                        logger.error("Too many consecutive failures. Device may be disconnected.")
                        break

                    time.sleep(0.5)
                    continue

                # Reset failure counter on success
                self.consecutive_failures = 0
                self.total_frames += 1

                # Process frame
                processed_frame, regions = self.process_frame(frame)

                # Calculate FPS (rolling average)
                self.frame_times.append(time.time())
                if len(self.frame_times) > 1:
                    fps = (len(self.frame_times) - 1) / (
                        self.frame_times[-1] - self.frame_times[0]
                    )
                else:
                    fps = 0.0

                # Draw overlay
                display_frame = self.draw_overlay(processed_frame, fps, len(regions))

                # Display
                cv2.imshow(window_name, display_frame)

                # Handle keyboard input with frame rate control
                wait_time = int(target_interval * 1000)
                key = cv2.waitKey(wait_time) & 0xFF

                if key != 255:  # Key pressed
                    action = self.handle_keyboard(key)
                    if action == 'quit':
                        self.running = False
                    elif action == 'screenshot':
                        self.save_screenshot(display_frame)

                # Maintain frame rate
                elapsed = time.time() - loop_start
                if elapsed < target_interval:
                    time.sleep(target_interval - elapsed)

        except KeyboardInterrupt:
            logger.info("\nInterrupted by user")
        except Exception as e:
            logger.error(f"Error in monitor loop: {e}", exc_info=True)
        finally:
            self.cleanup()

    def cleanup(self) -> None:
        """Release resources and cleanup."""
        if self.capture:
            self.capture.release()

        cv2.destroyAllWindows()

        logger.info("=" * 70)
        logger.info(f"Monitor stopped")
        logger.info(f"Total frames: {self.total_frames}")
        if self.detection_enabled:
            logger.info(f"Total detections: {self.total_detections}")
        logger.info("=" * 70)


def main():
    """Main entry point for stream monitor."""
    parser = argparse.ArgumentParser(
        description="HDMI Capture Stream Monitor with YOLO Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic streaming
  python -m automation.monitor_stream

  # With detection enabled
  python -m automation.monitor_stream --detection-on

  # Custom FPS and confidence
  python -m automation.monitor_stream --fps 10 --confidence 0.3

  # Debug mode
  python -m automation.monitor_stream --debug

Keyboard Controls:
  Q/ESC - Quit application
  D     - Toggle YOLO detection
  S     - Save screenshot
  F     - Toggle FPS counter
  H     - Toggle help overlay
  +     - Increase confidence threshold
  -     - Decrease confidence threshold
        """
    )

    # Device options
    parser.add_argument(
        '--device', type=int,
        help='Video capture device index (default: from config)'
    )
    parser.add_argument(
        '--fps', type=float, default=5.0,
        help='Target frame rate (default: 5.0)'
    )

    # Detection options
    parser.add_argument(
        '--detection-on', action='store_true',
        help='Enable YOLO detection from start'
    )
    parser.add_argument(
        '--confidence', type=float,
        help='Detection confidence threshold (default: from config)'
    )
    parser.add_argument(
        '--imgsz', type=int,
        help='YOLO image size (default: from config)'
    )

    # Display options
    parser.add_argument(
        '--no-help', action='store_true',
        help='Hide help overlay'
    )
    parser.add_argument(
        '--output-dir', type=str,
        help='Screenshot output directory (default: ./monitor_outputs)'
    )

    # Debug options
    parser.add_argument(
        '--debug', action='store_true',
        help='Enable debug logging'
    )
    parser.add_argument(
        '--env-file', type=str,
        help='Path to .env file'
    )

    args = parser.parse_args()

    # Load configuration
    try:
        config = load_config(args.env_file)
    except ValueError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        return 1

    # Apply argument overrides
    if args.device is not None:
        config.capture_device_index = args.device
    else:
        args.device = config.capture_device_index

    if args.confidence is not None:
        config.detection_confidence = args.confidence
    else:
        args.confidence = config.detection_confidence

    if args.imgsz is not None:
        config.detection_image_size = args.imgsz
    else:
        args.imgsz = config.detection_image_size

    if args.output_dir is not None:
        args.output_dir = Path(args.output_dir)
    else:
        args.output_dir = Path("./monitor_outputs")

    # Setup logging
    log_level = "DEBUG" if args.debug else config.log_level
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = config.log_dir / f"monitor_{timestamp}.log"

    setup_logging(log_level, log_file, debug_mode=args.debug)

    logger.info("HDMI Capture Stream Monitor")
    logger.info(f"Configuration: device={args.device}, fps={args.fps}, "
                f"detection={args.detection_on}, confidence={args.confidence}")

    # Run monitor
    try:
        monitor = StreamMonitor(config, args)
        monitor.run()
        return 0
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
