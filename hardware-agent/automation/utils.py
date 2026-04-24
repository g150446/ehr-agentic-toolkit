"""
Utility functions for Windows login automation.

Provides logging setup, debug utilities, and image handling functions.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
import cv2
import numpy as np


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[Path] = None,
    debug_mode: bool = False
) -> logging.Logger:
    """
    Set up logging configuration.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Path to log file. If None, logs to console only
        debug_mode: If True, sets level to DEBUG and adds detailed formatting

    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger("windows_login")
    logger.setLevel(logging.DEBUG if debug_mode else log_level)

    # Remove existing handlers
    logger.handlers.clear()

    # Create formatters
    if debug_mode:
        # Detailed format for debug mode
        formatter = logging.Formatter(
            '[%(asctime)s.%(msecs)03d] [%(name)s] [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    else:
        # Simple format for normal mode
        formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] %(message)s',
            datefmt='%H:%M:%S'
        )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.info(f"Logging to file: {log_file}")

    return logger


def debug_pause(message: str, debug_mode: bool = True) -> None:
    """
    Pause execution and wait for user input (only in debug mode).

    Args:
        message: Message to display to user
        debug_mode: If True, pauses and waits for Enter key
    """
    if debug_mode:
        input(f"\n[DEBUG PAUSE] {message} (Press Enter to continue): ")


def save_debug_image(
    image: np.ndarray,
    filename: str,
    output_dir: Path,
    timestamp: bool = True
) -> Path:
    """
    Save image for debugging purposes.

    Args:
        image: Image array to save
        filename: Base filename (e.g., "capture.jpg")
        output_dir: Directory to save image
        timestamp: If True, adds timestamp to filename

    Returns:
        Path to saved image file
    """
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Add timestamp if requested
    if timestamp:
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        name_parts = filename.rsplit('.', 1)
        if len(name_parts) == 2:
            filename = f"{name_parts[0]}_{timestamp_str}.{name_parts[1]}"
        else:
            filename = f"{filename}_{timestamp_str}"

    # Save image
    output_path = output_dir / filename
    cv2.imwrite(str(output_path), image)

    logger = logging.getLogger("windows_login")
    logger.debug(f"Saved debug image: {output_path}")

    return output_path


def draw_bounding_boxes(
    image: np.ndarray,
    boxes: list,
    labels: Optional[list] = None,
    scores: Optional[list] = None,
    color: tuple = (0, 255, 0),
    thickness: int = 2
) -> np.ndarray:
    """
    Draw bounding boxes on image.

    Args:
        image: Input image
        boxes: List of bounding boxes in format [x1, y1, x2, y2]
        labels: Optional list of labels for each box
        scores: Optional list of confidence scores for each box
        color: Box color in BGR format
        thickness: Line thickness

    Returns:
        Image with drawn bounding boxes
    """
    result = image.copy()

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)

        # Draw rectangle
        cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)

        # Add label and score if provided
        if labels or scores:
            label_text = ""
            if labels and i < len(labels):
                label_text = str(labels[i])
            if scores and i < len(scores):
                score_text = f"{scores[i]:.2f}"
                label_text = f"{label_text} {score_text}" if label_text else score_text

            if label_text:
                # Draw label background
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                font_thickness = 1
                (text_width, text_height), _ = cv2.getTextSize(
                    label_text, font, font_scale, font_thickness
                )

                cv2.rectangle(
                    result,
                    (x1, y1 - text_height - 6),
                    (x1 + text_width + 4, y1),
                    color,
                    -1
                )

                # Draw label text
                cv2.putText(
                    result,
                    label_text,
                    (x1 + 2, y1 - 4),
                    font,
                    font_scale,
                    (255, 255, 255),
                    font_thickness
                )

    return result


def calculate_center(box: list) -> dict:
    """
    Calculate center point of bounding box.

    Args:
        box: Bounding box in format [x1, y1, x2, y2]

    Returns:
        Dictionary with 'x' and 'y' center coordinates
    """
    x1, y1, x2, y2 = box
    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)
    return {'x': center_x, 'y': center_y}


def is_point_in_box(point: dict, box: list) -> bool:
    """
    Check if a point is inside a bounding box.

    Args:
        point: Dictionary with 'x' and 'y' coordinates
        box: Bounding box in format [x1, y1, x2, y2]

    Returns:
        True if point is inside box, False otherwise
    """
    x1, y1, x2, y2 = box
    return x1 <= point['x'] <= x2 and y1 <= point['y'] <= y2


def format_timestamp(dt: Optional[datetime] = None) -> str:
    """
    Format datetime as string for filenames.

    Args:
        dt: Datetime object. If None, uses current time.

    Returns:
        Formatted timestamp string
    """
    if dt is None:
        dt = datetime.now()
    return dt.strftime("%Y%m%d_%H%M%S")


class ProgressLogger:
    """Simple progress logger for multi-step operations."""

    def __init__(self, total_steps: int, logger: Optional[logging.Logger] = None):
        """
        Initialize progress logger.

        Args:
            total_steps: Total number of steps
            logger: Logger instance. If None, uses default logger
        """
        self.total_steps = total_steps
        self.current_step = 0
        self.logger = logger or logging.getLogger("windows_login")

    def step(self, message: str) -> None:
        """
        Log a step completion.

        Args:
            message: Step description
        """
        self.current_step += 1
        progress = (self.current_step / self.total_steps) * 100
        self.logger.info(f"[{self.current_step}/{self.total_steps}] ({progress:.0f}%) {message}")

    def complete(self, message: str = "All steps completed") -> None:
        """
        Log completion of all steps.

        Args:
            message: Completion message
        """
        self.logger.info(f"[COMPLETE] {message}")
