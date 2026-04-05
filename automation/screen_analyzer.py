"""
Screen Analyzer module for Windows login automation.

Handles screen capture, DocLayout-YOLO detection, and OCR text extraction
to locate password fields and other UI elements.
"""

import cv2
import logging
import numpy as np
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
import ssl

# Fix SSL certificate issue for EasyOCR model download
ssl._create_default_https_context = ssl._create_unverified_context


logger = logging.getLogger(__name__)


@dataclass
class DetectedRegion:
    """Represents a detected UI region from YOLO."""
    index: int
    class_name: str
    confidence: float
    bbox: Dict[str, int]  # {'x1', 'y1', 'x2', 'y2'}
    center: Dict[str, int]  # {'x', 'y'}
    size: Dict[str, int]  # {'width', 'height'}


@dataclass
class OCRResult:
    """Represents OCR text extraction result."""
    text: str
    confidence: float
    bbox: List[List[int]]  # Polygon coordinates
    region_index: int


@dataclass
class PasswordField:
    """Represents a detected password input field."""
    region: DetectedRegion
    ocr_matches: List[OCRResult]
    detection_method: str  # 'ocr_text', 'spatial', 'ui_class', 'manual'


def capture_screen(device_index: int = 0, width: int = 1920, height: int = 1080) -> Optional[np.ndarray]:
    """
    Capture screen from HDMI capture device.

    Args:
        device_index: Video capture device index (0 for MiraBox)
        width: Capture width in pixels
        height: Capture height in pixels

    Returns:
        Captured frame as numpy array, or None if capture failed
    """
    try:
        logger.info(f"Opening video capture device {device_index}...")
        cap = cv2.VideoCapture(device_index)

        if not cap.isOpened():
            logger.error(f"Failed to open capture device {device_index}")
            return None

        # Set capture resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        # Read frame
        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            logger.error("Failed to read frame from capture device")
            return None

        logger.info(f"Captured frame: {frame.shape[1]}x{frame.shape[0]} pixels")
        return frame

    except Exception as e:
        logger.error(f"Screen capture error: {e}")
        return None


def load_yolo_model(model_path: str, device: str = "auto"):
    """
    Load DocLayout-YOLO model.

    Args:
        model_path: Path to YOLO model file (.pt)
        device: Device to use ('cpu', 'cuda', 'mps', 'auto')

    Returns:
        Loaded YOLO model
    """
    try:
        from doclayout_yolo import YOLOv10

        logger.info(f"Loading YOLO model from {model_path}...")

        # Auto-detect device
        if device == "auto":
            import torch
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
            logger.info(f"Auto-selected device: {device}")

        model = YOLOv10(model_path)
        model.to(device)  # Explicitly set device
        logger.info(f"YOLO model loaded successfully on {device}")
        return model

    except Exception as e:
        logger.error(f"Failed to load YOLO model: {e}")
        raise


_ocr_reader_cache: dict = {}


def load_ocr_reader(languages: List[str] = ['ja', 'en'], use_gpu: bool = False):
    """
    Load EasyOCR reader (cached — subsequent calls with same args return the same instance).

    Args:
        languages: List of language codes
        use_gpu: Whether to use GPU for OCR (set True to use MPS on Apple Silicon)

    Returns:
        EasyOCR reader instance
    """
    cache_key = ('easyocr', tuple(languages), use_gpu)
    if cache_key in _ocr_reader_cache:
        logger.debug("Returning cached EasyOCR reader")
        return _ocr_reader_cache[cache_key]

    try:
        import easyocr

        logger.info(f"Loading EasyOCR reader (languages: {', '.join(languages)}, GPU: {use_gpu})...")
        reader = easyocr.Reader(languages, gpu=use_gpu)
        logger.info("EasyOCR reader loaded successfully")
        _ocr_reader_cache[cache_key] = reader
        return reader

    except Exception as e:
        logger.error(f"Failed to load OCR reader: {e}")
        raise


def load_rapidocr_reader():
    """
    Load RapidOCR reader with Japanese model (ONNX Runtime backend).

    Uses PP-OCRv4 Japan model — faster than EasyOCR on CPU/M1 via ONNX Runtime.
    Result is cached so repeated calls are instant.

    Returns:
        RapidOCR engine instance
    """
    cache_key = ('rapidocr',)
    if cache_key in _ocr_reader_cache:
        logger.debug("Returning cached RapidOCR reader")
        return _ocr_reader_cache[cache_key]

    try:
        from rapidocr import EngineType, LangRec, ModelType, OCRVersion, RapidOCR

        logger.info("Loading RapidOCR (ONNX, PP-OCRv4 Japan)...")
        engine = RapidOCR(params={
            "Rec.ocr_version": OCRVersion.PPOCRV4,
            "Rec.engine_type": EngineType.ONNXRUNTIME,
            "Rec.lang_type": LangRec.JAPAN,
            "Rec.model_type": ModelType.MOBILE,
        })
        logger.info("RapidOCR loaded successfully")
        _ocr_reader_cache[cache_key] = engine
        return engine

    except Exception as e:
        logger.error(f"Failed to load RapidOCR: {e}")
        raise


def run_ocr_word_split(reader, image: np.ndarray, gap_ratio: float = 1.5) -> List[tuple]:
    """
    Run RapidOCR with character-level boxes, then split merged lines into separate
    segments wherever horizontal gap between characters exceeds gap_ratio × avg char width.

    This solves the case where a horizontal menu bar like
    "受付患者一覧　予約患者一覧　枠別予約患者一覧" is returned as one segment.
    Each menu item is returned as a separate (bbox, text, confidence) tuple.

    Only supported for the RapidOCR backend.  Falls back to run_ocr() for EasyOCR.

    Args:
        reader: RapidOCR engine instance
        image: Input image as numpy array (BGR)
        gap_ratio: Gap threshold multiplier — split when gap > gap_ratio × avg_char_width

    Returns:
        List of (bbox, text, confidence) tuples, one per detected word/phrase segment
    """
    if type(reader).__name__ == 'Reader':
        # EasyOCR — fall back to standard readtext
        return reader.readtext(image)

    import cv2 as _cv2
    image_rgb = _cv2.cvtColor(image, _cv2.COLOR_BGR2RGB)
    result = reader(image_rgb, return_word_box=True)

    if result is None or result.word_results is None:
        return []

    output = []
    for line_chars in result.word_results:
        if not line_chars:
            continue

        # Compute average character width for this line
        char_widths = []
        for char, conf, box in line_chars:
            x_coords = [p[0] for p in box]
            char_widths.append(max(x_coords) - min(x_coords))
        avg_w = sum(char_widths) / len(char_widths) if char_widths else 1

        # Group characters into segments by gap
        segments = []   # each segment: list of (char, conf, box)
        current = [line_chars[0]]

        for i in range(1, len(line_chars)):
            prev_box = line_chars[i - 1][2]
            curr_box = line_chars[i][2]
            prev_x2 = max(p[0] for p in prev_box)
            curr_x1 = min(p[0] for p in curr_box)
            gap = curr_x1 - prev_x2

            if gap > gap_ratio * avg_w:
                segments.append(current)
                current = [line_chars[i]]
            else:
                current.append(line_chars[i])

        segments.append(current)

        # Convert each segment to (bbox, text, confidence)
        for seg in segments:
            text = ''.join(c[0] for c in seg)
            conf = sum(c[1] for c in seg) / len(seg)

            all_points = [p for c in seg for p in c[2]]
            x_min = min(p[0] for p in all_points)
            y_min = min(p[1] for p in all_points)
            x_max = max(p[0] for p in all_points)
            y_max = max(p[1] for p in all_points)
            bbox = [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]

            output.append((bbox, text, float(conf)))

    return output


def run_ocr(reader, image: np.ndarray) -> List[tuple]:
    """
    Run OCR on an image, normalizing output to EasyOCR-compatible format.

    Args:
        reader: EasyOCR reader or RapidOCR engine instance
        image: Input image as numpy array (BGR)

    Returns:
        List of (bbox, text, confidence) tuples where bbox is [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
    """
    # Detect backend by class name to avoid importing both
    reader_type = type(reader).__name__

    if reader_type == 'Reader':
        # EasyOCR
        return reader.readtext(image)

    # RapidOCR
    import cv2 as _cv2
    image_rgb = _cv2.cvtColor(image, _cv2.COLOR_BGR2RGB)
    result = reader(image_rgb)
    if result is None or result.boxes is None:
        return []
    output = []
    for box, text, score in zip(result.boxes, result.txts, result.scores):
        # box is ndarray shape (4, 2) — same layout as EasyOCR bbox
        output.append((box.tolist(), text, float(score) if score is not None else 0.0))
    return output


def analyze_layout(
    image: np.ndarray,
    model,
    confidence: float = 0.2,
    image_size: int = 1024
) -> List[DetectedRegion]:
    """
    Analyze image layout using YOLO detection.

    Args:
        image: Input image as numpy array
        model: Loaded YOLO model
        confidence: Confidence threshold for detection
        image_size: Input image size for YOLO

    Returns:
        List of detected regions
    """
    try:
        logger.debug(f"Running YOLO detection (conf={confidence}, imgsz={image_size})...")

        # Perform prediction (model already loaded on correct device)
        det_res = model.predict(
            image,
            imgsz=image_size,
            conf=confidence
        )

        # Extract detected regions
        regions = []
        for i, box in enumerate(det_res[0].boxes):
            class_id = int(box.cls[0])
            conf = float(box.conf[0])
            coords = box.xyxy[0].tolist()
            class_name = model.names[class_id]

            # Get coordinates
            x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
            width = x2 - x1
            height = y2 - y1
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            region = DetectedRegion(
                index=i,
                class_name=class_name,
                confidence=conf,
                bbox={'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2},
                center={'x': center_x, 'y': center_y},
                size={'width': width, 'height': height}
            )
            regions.append(region)

        logger.info(f"Detected {len(regions)} regions")
        for region in regions:
            logger.debug(
                f"  Region {region.index}: {region.class_name} "
                f"at ({region.center['x']}, {region.center['y']}) "
                f"conf={region.confidence:.2f}"
            )

        return regions

    except Exception as e:
        logger.error(f"Layout analysis error: {e}")
        return []


def extract_text(
    image: np.ndarray,
    regions: List[DetectedRegion],
    ocr_reader
) -> List[OCRResult]:
    """
    Extract text from detected regions using OCR.

    Args:
        image: Original image
        regions: List of detected regions
        ocr_reader: EasyOCR reader instance

    Returns:
        List of OCR results
    """
    ocr_results = []

    try:
        for region in regions:
            # Crop region
            x1, y1 = region.bbox['x1'], region.bbox['y1']
            x2, y2 = region.bbox['x2'], region.bbox['y2']
            cropped = image[y1:y2, x1:x2]

            if cropped.size == 0:
                continue

            # Perform OCR
            logger.debug(f"Running OCR on region {region.index} ({region.class_name})...")
            results = ocr_reader.readtext(cropped)

            for result in results:
                bbox, text, conf = result
                ocr_results.append(OCRResult(
                    text=text,
                    confidence=conf,
                    bbox=bbox,
                    region_index=region.index
                ))
                logger.debug(f"  OCR: '{text}' (conf={conf:.2f})")

    except Exception as e:
        logger.error(f"OCR extraction error: {e}")

    logger.info(f"Extracted {len(ocr_results)} text segments")
    return ocr_results


def find_password_field(
    regions: List[DetectedRegion],
    ocr_results: List[OCRResult],
    image_shape: tuple
) -> Optional[PasswordField]:
    """
    Find password input field using multiple strategies.

    Args:
        regions: List of detected regions
        ocr_results: List of OCR results
        image_shape: Image dimensions (height, width, channels)

    Returns:
        PasswordField if found, None otherwise
    """
    logger.debug("Searching for password field...")

    # Strategy 1: OCR text matching
    password_keywords = ['password', 'パスワード', 'pass', 'pwd', ' password:', 'パスワード:']
    password_keywords_lower = [k.lower() for k in password_keywords]

    for ocr_result in ocr_results:
        text_lower = ocr_result.text.lower()
        if any(keyword in text_lower for keyword in password_keywords_lower):
            # Find the region containing this OCR result
            for region in regions:
                if region.index == ocr_result.region_index:
                    logger.info(
                        f"Password field found via OCR text matching: "
                        f"'{ocr_result.text}' in region {region.index}"
                    )
                    return PasswordField(
                        region=region,
                        ocr_matches=[ocr_result],
                        detection_method='ocr_text'
                    )

    # Strategy 2: Spatial positioning (center of screen, below username)
    image_height, image_width = image_shape[:2]
    center_x = image_width // 2
    center_y = image_height // 2

    # Look for regions near center
    center_regions = []
    for region in regions:
        dx = abs(region.center['x'] - center_x)
        dy = abs(region.center['y'] - center_y)

        # Within 30% of center
        if dx < image_width * 0.3 and dy < image_height * 0.3:
            distance = (dx**2 + dy**2)**0.5
            center_regions.append((distance, region))

    if center_regions:
        # Sort by distance to center
        center_regions.sort(key=lambda x: x[0])

        # Prefer regions in lower half (where password typically is)
        for _, region in center_regions:
            if region.center['y'] > center_y:
                logger.info(
                    f"Password field found via spatial positioning: "
                    f"region {region.index} at ({region.center['x']}, {region.center['y']})"
                )
                return PasswordField(
                    region=region,
                    ocr_matches=[],
                    detection_method='spatial'
                )

    # Strategy 3: Sort regions spatially and pick middle-lower region
    sorted_regions = sorted(regions, key=lambda x: (x['center']['y'], x['center']['x']))
    if len(sorted_regions) > 0:
        # Pick region in lower-middle portion
        middle_idx = len(sorted_regions) // 2
        candidate = sorted_regions[middle_idx]

        logger.warning(
            f"Password field detection fallback: using region {candidate.index} "
            f"(sorted position {middle_idx}/{len(sorted_regions)})"
        )
        return PasswordField(
            region=candidate,
            ocr_matches=[],
            detection_method='spatial'
        )

    logger.warning("Password field not found - no suitable regions detected")
    return None


def sort_regions_spatially(regions: List[DetectedRegion]) -> List[DetectedRegion]:
    """
    Sort regions by spatial position (top to bottom, left to right).

    Args:
        regions: List of detected regions

    Returns:
        Sorted list of regions
    """
    return sorted(regions, key=lambda x: (x.center['y'], x.center['x']))


def visualize_detections(
    image: np.ndarray,
    regions: List[DetectedRegion],
    password_field: Optional[PasswordField] = None
) -> np.ndarray:
    """
    Visualize detected regions on image.

    Args:
        image: Input image
        regions: List of detected regions
        password_field: Detected password field (highlighted if provided)

    Returns:
        Annotated image
    """
    result = image.copy()

    for region in regions:
        x1, y1 = region.bbox['x1'], region.bbox['y1']
        x2, y2 = region.bbox['x2'], region.bbox['y2']

        # Highlight password field in red, others in green
        if password_field and region.index == password_field.region.index:
            color = (0, 0, 255)  # Red
            thickness = 4
        else:
            color = (0, 255, 0)  # Green
            thickness = 2

        # Draw rectangle
        cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)

        # Draw label
        label = f"#{region.index}: {region.class_name} ({region.confidence:.2f})"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 2

        # Background for label
        (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, font_thickness)
        cv2.rectangle(
            result,
            (x1, y1 - text_height - 10),
            (x1 + text_width + 10, y1),
            color,
            -1
        )

        # Label text
        cv2.putText(
            result,
            label,
            (x1 + 5, y1 - 5),
            font,
            font_scale,
            (255, 255, 255),
            font_thickness
        )

        # Draw center point
        center_x, center_y = region.center['x'], region.center['y']
        cv2.circle(result, (center_x, center_y), 5, color, -1)

    return result
