from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = BASE_DIR / "roadguard_best.pt"
EXPECTED_CLASSES = {
    0: "Longitudinal Crack",
    1: "Transverse Crack",
    2: "Alligator Crack",
    3: "Other Corruption",
    4: "Pothole",
}


def _image_array(image: Any) -> np.ndarray:
    array = np.asarray(image)
    if array.size == 0:
        raise ValueError("Input image is empty.")
    if array.ndim == 2:
        array = np.repeat(array[:, :, None], 3, axis=2)
    elif array.ndim == 3 and array.shape[2] == 4:
        array = array[:, :, :3]
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("Expected an RGB, RGBA, or grayscale image.")
    return np.clip(array, 0, 255).astype(np.uint8)


def preprocess_image(image: Any, enhance: bool = True) -> np.ndarray:
    """Improve contrast without changing image dimensions or box coordinates."""
    rgb = _image_array(image)
    if not enhance:
        return rgb
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    lightness, a_channel, b_channel = cv2.split(lab)
    lightness = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lightness)
    enhanced = cv2.cvtColor(cv2.merge((lightness, a_channel, b_channel)), cv2.COLOR_LAB2RGB)
    return enhanced


def load_model(model_path: Optional[Union[str, Path]] = None) -> YOLO:
    path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
    path = path if path.is_absolute() else BASE_DIR / path
    if not path.is_file():
        raise FileNotFoundError(f"Model file not found: {path}")
    return YOLO(str(path.resolve()))


def model_class_names(model: YOLO) -> Dict[int, str]:
    names = getattr(model, "names", None)
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    if isinstance(names, list):
        return {index: str(value) for index, value in enumerate(names)}
    return EXPECTED_CLASSES.copy()


def detect_road_damage(
    image: Any,
    model: Optional[YOLO] = None,
    model_path: Optional[Union[str, Path]] = None,
    confidence: float = 0.35,
    iou: float = 0.45,
    image_size: int = 640,
    enhance: bool = True,
) -> List[Dict[str, Any]]:
    """Run real YOLO inference and return every detected road-damage class."""
    if not 0.0 <= float(confidence) <= 1.0:
        raise ValueError("confidence must be between 0 and 1")
    if not 0.0 <= float(iou) <= 1.0:
        raise ValueError("iou must be between 0 and 1")
    if int(image_size) <= 0:
        raise ValueError("image_size must be positive")

    original = _image_array(image)
    prepared = preprocess_image(original, enhance=enhance)
    detector = model or load_model(model_path)
    names = model_class_names(detector)
    results = detector.predict(
        source=prepared,
        conf=float(confidence),
        iou=float(iou),
        imgsz=int(image_size),
        verbose=False,
    )
    if not results or results[0].boxes is None:
        return []

    boxes = results[0].boxes
    detections: List[Dict[str, Any]] = []
    for index in range(len(boxes)):
        class_id = int(boxes.cls[index].item())
        score = float(boxes.conf[index].item())
        x1, y1, x2, y2 = [float(v) for v in boxes.xyxy[index].tolist()]
        x1 = max(0.0, min(x1, original.shape[1]))
        x2 = max(0.0, min(x2, original.shape[1]))
        y1 = max(0.0, min(y1, original.shape[0]))
        y2 = max(0.0, min(y2, original.shape[0]))
        detections.append({
            "id": index + 1,
            "class_id": class_id,
            "class_name": names.get(class_id, EXPECTED_CLASSES.get(class_id, f"Class {class_id}")),
            "confidence": score,
            "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "area": max(0.0, x2 - x1) * max(0.0, y2 - y1),
        })
    return detections


def validate_model_classes(model_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Report the classes embedded in the weights; never silently relabel them."""
    model = load_model(model_path)
    names = model_class_names(model)
    return {
        "model_path": str(Path(model_path or DEFAULT_MODEL_PATH).resolve()),
        "classes": names,
        "expected_classes": EXPECTED_CLASSES,
        "class_count": len(names),
        "has_pothole_class": any("pothole" in name.lower() for name in names.values()),
    }
