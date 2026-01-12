"""
Model Manager for YOLO Model Switching

This module provides a unified interface for switching between multiple YOLO models:
- DocLayout-YOLO: Document layout detection
- YOLOv11: UI element detection (foduucom/web-form-ui-field-detection)

The ModelManager handles lazy loading, model switching, and provides a unified
detection interface that abstracts away model-specific differences.
"""

import logging
from enum import Enum
from typing import List, Tuple, Optional
import numpy as np

logger = logging.getLogger("model_manager")


class ModelType(Enum):
    """Supported YOLO model types"""
    DOCLAYOUT = "doclayout"
    UI_DETECTION = "ui-detection"


class DetectionResult:
    """
    Unified detection result format for all YOLO models.

    Attributes:
        bbox: Bounding box coordinates (x1, y1, x2, y2)
        confidence: Detection confidence score (0.0-1.0)
        label: Detected class label
    """

    def __init__(self, bbox: Tuple[int, int, int, int], confidence: float, label: str):
        self.bbox = bbox  # (x1, y1, x2, y2)
        self.confidence = confidence
        self.label = label

    def __repr__(self) -> str:
        return f"DetectionResult(label='{self.label}', confidence={self.confidence:.2f}, bbox={self.bbox})"


class ModelManager:
    """
    Manages multiple YOLO models with unified interface.

    Provides lazy loading, model switching, and unified detection interface
    that works with both DocLayout-YOLO and YOLOv11 models.
    """

    def __init__(self, config):
        """
        Initialize Model Manager.

        Args:
            config: AutomationConfig object containing model paths and settings
        """
        self.config = config
        self.current_model_type = ModelType.DOCLAYOUT
        self.doclayout_model = None
        self.ui_detection_model = None

        logger.info("Model Manager initialized")

    def load_doclayout_model(self):
        """
        Load DocLayout-YOLO model (lazy loading).

        Returns:
            Loaded DocLayout-YOLO model instance
        """
        if self.doclayout_model is None:
            logger.info("Loading DocLayout-YOLO model...")
            try:
                from automation.screen_analyzer import load_yolo_model
                self.doclayout_model = load_yolo_model(
                    self.config.doclayout_model_path,
                    self.config.detection_device
                )
                logger.info("DocLayout-YOLO model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load DocLayout-YOLO model: {e}")
                raise

        return self.doclayout_model

    def load_ui_detection_model(self):
        """
        Load YOLOv8 UI detection model from HuggingFace (lazy loading).

        Uses ultralyticsplus to load the foduucom/web-form-ui-field-detection model.

        Returns:
            Loaded YOLOv8 model instance
        """
        if self.ui_detection_model is None:
            logger.info("Loading YOLOv8 UI detection model from HuggingFace...")
            try:
                from ultralyticsplus import YOLO
                self.ui_detection_model = YOLO('foduucom/web-form-ui-field-detection')

                # Set model parameters for optimal performance
                self.ui_detection_model.overrides['conf'] = 0.25  # NMS confidence threshold
                self.ui_detection_model.overrides['iou'] = 0.45   # NMS IoU threshold
                self.ui_detection_model.overrides['agnostic_nms'] = False
                self.ui_detection_model.overrides['max_det'] = 1000

                logger.info("YOLOv8 UI detection model loaded successfully")
            except ImportError:
                logger.error("ultralyticsplus not installed. Install with: pip install ultralyticsplus")
                raise
            except Exception as e:
                logger.error(f"Failed to load YOLOv8 UI detection model: {e}")
                raise

        return self.ui_detection_model

    def switch_model(self, model_type: ModelType):
        """
        Switch active YOLO model.

        Args:
            model_type: Target model type to switch to
        """
        self.current_model_type = model_type
        logger.info(f"Switched to {model_type.value} model")

    def detect(self, image: np.ndarray, confidence: float = 0.2) -> List[DetectionResult]:
        """
        Unified detection interface for all models.

        Performs object detection using the currently active model and returns
        results in a unified format regardless of the underlying model.

        Args:
            image: Input image as numpy array (BGR format)
            confidence: Minimum confidence threshold for detections

        Returns:
            List of DetectionResult objects
        """
        if self.current_model_type == ModelType.DOCLAYOUT:
            model = self.load_doclayout_model()
            results = model.predict(
                image,
                imgsz=self.config.detection_image_size,
                conf=confidence,
                verbose=False
            )
        else:
            model = self.load_ui_detection_model()
            results = model.predict(
                image,
                conf=confidence,
                verbose=False
            )

        # Convert to unified format
        detections = []

        for result in results:
            boxes = result.boxes
            for i in range(len(boxes)):
                # Extract bbox coordinates
                bbox = boxes.xyxy[i].cpu().numpy().astype(int)

                # Extract confidence
                conf = float(boxes.conf[i])

                # Extract class label
                cls_id = int(boxes.cls[i])
                label = result.names[cls_id]

                detections.append(DetectionResult(
                    bbox=tuple(bbox),
                    confidence=conf,
                    label=label
                ))

        logger.debug(f"Detected {len(detections)} objects with {self.current_model_type.value} model")
        return detections

    def get_current_model_name(self) -> str:
        """
        Get the name of the currently active model.

        Returns:
            Model name as string
        """
        return self.current_model_type.value

    def get_current_model_type(self) -> ModelType:
        """
        Get the currently active model type.

        Returns:
            Current ModelType enum value
        """
        return self.current_model_type
