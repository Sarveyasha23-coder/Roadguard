```python
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "roadguard_best.pt"

CLASS_NAMES = {
    0: "Longitudinal Crack",
    1: "Transverse Crack",
    2: "Alligator Crack",
    3: "Other Corruption",
    4: "Pothole",
}


# ============================================================
# MODEL
# ============================================================

_model = None


def load_model():
    """Load the YOLO model once and reuse it."""
    global _model

    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model file not found: {MODEL_PATH}"
            )

        _model = YOLO(str(MODEL_PATH))

    return _model


# ============================================================
# VALIDATION
# ============================================================

def validate_image(image: np.ndarray) -> None:
    """Validate input image."""

    if image is None:
        raise ValueError(
            "Image is empty or could not be decoded."
        )

    if not isinstance(image, np.ndarray):
        raise TypeError(
            "Image must be a NumPy array."
        )

    if image.size == 0:
        raise ValueError(
            "Image contains no data."
        )


# ============================================================
# ROAD DAMAGE DETECTION
# ============================================================

def detect_road_damage(
    image: np.ndarray,
    confidence: float = 0.25,
    iou: float = 0.45,
    image_size: int = 640,
) -> Dict[str, Any]:

    validate_image(image)

    model = load_model()

    height, width = image.shape[:2]

    results = model.predict(
        source=image,
        conf=confidence,
        iou=iou,
        imgsz=image_size,
        verbose=False,
    )

    result = results[0]

    detections: List[Dict[str, Any]] = []

    if result.boxes is not None:

        boxes = result.boxes

        xyxy = boxes.xyxy.cpu().numpy()
        confidences = boxes.conf.cpu().numpy()
        class_ids = boxes.cls.cpu().numpy().astype(int)

        for box, conf, class_id in zip(
            xyxy,
            confidences,
            class_ids,
        ):

            x1, y1, x2, y2 = box.tolist()

            damage_type = CLASS_NAMES.get(
                class_id,
                f"Unknown ({class_id})",
            )

            box_width = max(
                0.0,
                x2 - x1,
            )

            box_height = max(
                0.0,
                y2 - y1,
            )

            box_area = (
                box_width * box_height
            )

            image_area = (
                width * height
            )

            area_percentage = (
                (box_area / image_area) * 100
                if image_area > 0
                else 0
            )

            detections.append(
                {
                    "class_id": int(class_id),
                    "damage_type": damage_type,
                    "confidence": float(conf),
                    "confidence_percentage": round(
                        float(conf) * 100,
                        2,
                    ),
                    "x1": round(float(x1), 2),
                    "y1": round(float(y1), 2),
                    "x2": round(float(x2), 2),
                    "y2": round(float(y2), 2),
                    "width": round(
                        box_width,
                        2,
                    ),
                    "height": round(
                        box_height,
                        2,
                    ),
                    "area_percentage": round(
                        area_percentage,
                        2,
                    ),
                }
            )

    # YOLO annotated image
    annotated_image = result.plot()

    # Class statistics
    class_counts: Dict[str, int] = {}

    for detection in detections:

        damage_type = detection[
            "damage_type"
        ]

        class_counts[damage_type] = (
            class_counts.get(
                damage_type,
                0,
            )
            + 1
        )

    average_confidence = (
        sum(
            d["confidence"]
            for d in detections
        )
        / len(detections)
        if detections
        else 0.0
    )

    return {
        "detections": detections,
        "annotated_image": annotated_image,
        "image_width": width,
        "image_height": height,
        "detection_count": len(detections),
        "class_counts": class_counts,
        "average_confidence": round(
            average_confidence * 100,
            2,
        ),
    }


# ============================================================
# SIMPLE API
# ============================================================

def analyze_image(
    image: np.ndarray,
    confidence: float = 0.25,
) -> Dict[str, Any]:

    return detect_road_damage(
        image=image,
        confidence=confidence,
    )


# ============================================================
# COMMAND-LINE TEST
# ============================================================

if __name__ == "__main__":

    import sys

    if len(sys.argv) < 2:

        print(
            "Usage: python inference.py <image_path>"
        )

        raise SystemExit(1)

    image_path = sys.argv[1]

    image = cv2.imread(
        image_path
    )

    if image is None:

        print(
            f"Could not read image: {image_path}"
        )

        raise SystemExit(1)

    result = analyze_image(image)

    print("\nRoadGuard Detection Results")
    print("=" * 40)

    for detection in result[
        "detections"
    ]:

        print(
            f"{detection['damage_type']}: "
            f"{detection['confidence_percentage']}%"
        )

    print(
        f"\nTotal detections: "
        f"{result['detection_count']}"
    )
```
