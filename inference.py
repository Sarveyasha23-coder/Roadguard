"""
============================================================
ROADGUARD INFRASTRUCTURE INTELLIGENCE
Inference Engine
============================================================

Purpose:
    Load the trained YOLOv8n model and perform road-damage
    detection on images.

Dataset:
    RDD2022

Model:
    YOLOv8n

Expected model file:
    roadguard_best.pt

Expected project structure:

    Roadguard/
    ├── app.py
    ├── inference.py
    ├── severity.py
    ├── roadguard_best.pt
    ├── requirements.txt
    └── README.md

IMPORTANT:
    The model is intentionally loaded from the project root.
    Do NOT use "models/roadguard_best.pt" unless you create
    a models folder yourself.
============================================================
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "roadguard_best.pt"


# ============================================================
# ROAD DAMAGE CLASSES
# ============================================================

CLASS_NAMES = {
    0: "Longitudinal Crack",
    1: "Transverse Crack",
    2: "Alligator Crack",
    3: "Other Corruption",
    4: "Pothole",
}


# ============================================================
# ULTRALYTICS IMPORT
# ============================================================

try:
    from ultralytics import YOLO
except Exception as exc:
    raise RuntimeError(
        "\n"
        "============================================================\n"
        "ROADGUARD: ULTRALYTICS IMPORT ERROR\n"
        "============================================================\n"
        f"Error type: {type(exc).__name__}\n"
        f"Error message: {exc}\n"
        "\n"
        "Ultralytics is listed in requirements.txt, but the package\n"
        "could not be imported successfully.\n"
        "\n"
        "Check the Streamlit deployment logs for the full error.\n"
        "============================================================\n"
    ) from exc


# ============================================================
# MODEL LOADING
# ============================================================

_model: Optional[Any] = None


def get_model(
    model_path: Optional[Union[str, Path]] = None,
) -> Any:
    """
    Load and cache the YOLO model.

    Parameters
    ----------
    model_path:
        Optional path to a YOLO .pt model.

        If omitted, RoadGuard automatically uses:

            roadguard_best.pt

        from the project root.

    Returns
    -------
    YOLO
        Loaded Ultralytics YOLO model.
    """

    global _model

    if _model is not None:
        return _model

    if model_path is None:
        model_file = MODEL_PATH
    else:
        model_file = Path(model_path)

        if not model_file.is_absolute():
            model_file = BASE_DIR / model_file

    model_file = model_file.resolve()

    # --------------------------------------------------------
    # Check model existence
    # --------------------------------------------------------

    if not model_file.exists():
        raise FileNotFoundError(
            "\n"
            "============================================================\n"
            "ROADGUARD MODEL NOT FOUND\n"
            "============================================================\n"
            f"Expected model:\n{model_file}\n"
            "\n"
            "Make sure roadguard_best.pt exists in the project root.\n"
            "\n"
            "Expected structure:\n"
            "Roadguard/\n"
            "├── app.py\n"
            "├── inference.py\n"
            "├── severity.py\n"
            "├── roadguard_best.pt\n"
            "└── requirements.txt\n"
            "============================================================\n"
        )

    if not model_file.is_file():
        raise FileNotFoundError(
            f"Model path exists but is not a file:\n{model_file}"
        )

    # --------------------------------------------------------
    # Load YOLO model
    # --------------------------------------------------------

    try:
        _model = YOLO(str(model_file))
    except Exception as exc:
        raise RuntimeError(
            "\n"
            "============================================================\n"
            "ROADGUARD MODEL LOAD ERROR\n"
            "============================================================\n"
            f"Model path:\n{model_file}\n"
            "\n"
            f"Error type: {type(exc).__name__}\n"
            f"Error message: {exc}\n"
            "\n"
            "The model file exists, but Ultralytics could not load it.\n"
            "Make sure roadguard_best.pt is a valid YOLO model.\n"
            "============================================================\n"
        ) from exc

    return _model


# ============================================================
# CLASS NAME HELPER
# ============================================================

def get_class_name(class_id: int) -> str:
    """
    Convert a numeric class ID into a RoadGuard class name.
    """

    class_id = int(class_id)

    if class_id in CLASS_NAMES:
        return CLASS_NAMES[class_id]

    return f"Class {class_id}"


# ============================================================
# IMAGE VALIDATION
# ============================================================

def validate_image(image: Any) -> np.ndarray:
    """
    Validate and convert an input image into a NumPy RGB array.

    Supported inputs include:

    - PIL Image
    - NumPy array
    - Objects convertible to NumPy arrays

    Returns
    -------
    np.ndarray
        RGB image array.
    """

    if image is None:
        raise ValueError("Input image cannot be None.")

    try:
        image_array = np.asarray(image)
    except Exception as exc:
        raise ValueError(
            f"Unable to convert input image to NumPy array: {exc}"
        ) from exc

    if image_array.size == 0:
        raise ValueError("Input image is empty.")

    # --------------------------------------------------------
    # Grayscale image
    # --------------------------------------------------------

    if image_array.ndim == 2:
        image_array = np.stack(
            [image_array, image_array, image_array],
            axis=-1,
        )

    # --------------------------------------------------------
    # RGBA image
    # --------------------------------------------------------

    elif image_array.ndim == 3 and image_array.shape[2] == 4:
        image_array = image_array[:, :, :3]

    # --------------------------------------------------------
    # RGB image
    # --------------------------------------------------------

    elif image_array.ndim == 3 and image_array.shape[2] == 3:
        pass

    else:
        raise ValueError(
            "Unsupported image format. "
            "Expected an RGB, RGBA, or grayscale image."
        )

    # --------------------------------------------------------
    # Ensure uint8
    # --------------------------------------------------------

    if image_array.dtype != np.uint8:

        if np.issubdtype(image_array.dtype, np.floating):

            if image_array.max() <= 1.0:
                image_array = image_array * 255.0

        image_array = np.clip(
            image_array,
            0,
            255,
        ).astype(np.uint8)

    return image_array


# ============================================================
# SINGLE IMAGE INFERENCE
# ============================================================

def predict_image(
    image: Any,
    confidence: float = 0.25,
    iou: float = 0.45,
    image_size: int = 640,
    model_path: Optional[Union[str, Path]] = None,
    device: Optional[Union[str, int]] = None,
) -> List[Dict[str, Any]]:
    """
    Run road-damage detection on one image.

    Parameters
    ----------
    image:
        Input image.

    confidence:
        Minimum detection confidence.

    iou:
        IoU threshold used by YOLO NMS.

    image_size:
        YOLO inference image size.

    model_path:
        Optional custom model path.

    device:
        Optional inference device.

        Examples:
            "cpu"
            0
            "0"

    Returns
    -------
    list of dict
        Detection results.

    Example
    -------
    [
        {
            "class_id": 4,
            "class_name": "Pothole",
            "confidence": 0.91,
            "bbox": {
                "x1": 100,
                "y1": 120,
                "x2": 350,
                "y2": 300
            },
            "bbox_xyxy": [100, 120, 350, 300],
            "width": 250,
            "height": 180,
            "area": 45000
        }
    ]
    """

    # --------------------------------------------------------
    # Validate parameters
    # --------------------------------------------------------

    confidence = float(confidence)
    iou = float(iou)
    image_size = int(image_size)

    if not 0.0 <= confidence <= 1.0:
        raise ValueError(
            "confidence must be between 0.0 and 1.0."
        )

    if not 0.0 <= iou <= 1.0:
        raise ValueError(
            "iou must be between 0.0 and 1.0."
        )

    if image_size <= 0:
        raise ValueError(
            "image_size must be greater than zero."
        )

    # --------------------------------------------------------
    # Validate image
    # --------------------------------------------------------

    image_array = validate_image(image)

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = get_model(model_path)

    # --------------------------------------------------------
    # Run YOLO
    # --------------------------------------------------------

    prediction_kwargs = {
        "source": image_array,
        "conf": confidence,
        "iou": iou,
        "imgsz": image_size,
        "verbose": False,
    }

    if device is not None:
        prediction_kwargs["device"] = device

    try:
        results = model.predict(**prediction_kwargs)
    except Exception as exc:
        raise RuntimeError(
            "\n"
            "============================================================\n"
            "ROADGUARD INFERENCE ERROR\n"
            "============================================================\n"
            f"Error type: {type(exc).__name__}\n"
            f"Error message: {exc}\n"
            "============================================================\n"
        ) from exc

    if not results:
        return []

    result = results[0]

    detections: List[Dict[str, Any]] = []

    # --------------------------------------------------------
    # No bounding boxes
    # --------------------------------------------------------

    if result.boxes is None:
        return detections

    # --------------------------------------------------------
    # Extract boxes
    # --------------------------------------------------------

    boxes = result.boxes

    for index in range(len(boxes)):

        # Class ID
        class_id = int(
            boxes.cls[index].item()
        )

        # Confidence
        score = float(
            boxes.conf[index].item()
        )

        # Bounding box
        xyxy = boxes.xyxy[index].tolist()

        x1, y1, x2, y2 = [
            float(value)
            for value in xyxy
        ]

        width = max(
            0.0,
            x2 - x1,
        )

        height = max(
            0.0,
            y2 - y1,
        )

        area = width * height

        detections.append(
            {
                "class_id": class_id,
                "class_name": get_class_name(class_id),
                "confidence": score,

                "bbox": {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                },

                "bbox_xyxy": [
                    x1,
                    y1,
                    x2,
                    y2,
                ],

                "width": width,
                "height": height,
                "area": area,
            }
        )

    return detections


# ============================================================
# ALIAS
# ============================================================

def run_inference(
    image: Any,
    confidence: float = 0.25,
    iou: float = 0.45,
    image_size: int = 640,
    model_path: Optional[Union[str, Path]] = None,
    device: Optional[Union[str, int]] = None,
) -> List[Dict[str, Any]]:
    """
    Compatibility wrapper around predict_image().
    """

    return predict_image(
        image=image,
        confidence=confidence,
        iou=iou,
        image_size=image_size,
        model_path=model_path,
        device=device,
    )


# ============================================================
# ANNOTATED IMAGE
# ============================================================

def annotate_image(
    image: Any,
    detections: List[Dict[str, Any]],
    thickness: int = 3,
) -> np.ndarray:
    """
    Draw RoadGuard detections on an image.

    Returns
    -------
    np.ndarray
        Annotated RGB image.
    """

    image_array = validate_image(image).copy()

    # OpenCV is imported here so that the core model-loading
    # functionality remains easier to diagnose.
    try:
        import cv2
    except Exception as exc:
        raise RuntimeError(
            "OpenCV could not be imported: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    # --------------------------------------------------------
    # Draw detections
    # --------------------------------------------------------

    for detection in detections:

        bbox = detection.get("bbox", {})

        x1 = int(round(bbox.get("x1", 0)))
        y1 = int(round(bbox.get("y1", 0)))
        x2 = int(round(bbox.get("x2", 0)))
        y2 = int(round(bbox.get("y2", 0)))

        class_name = str(
            detection.get(
                "class_name",
                "Unknown",
            )
        )

        confidence_value = float(
            detection.get(
                "confidence",
                0.0,
            )
        )

        label = (
            f"{class_name} "
            f"{confidence_value:.2f}"
        )

        # ----------------------------------------------------
        # Bounding box
        # ----------------------------------------------------

        cv2.rectangle(
            image_array,
            (x1, y1),
            (x2, y2),
            (255, 0, 0),
            thickness,
        )

        # ----------------------------------------------------
        # Label background
        # ----------------------------------------------------

        font = cv2.FONT_HERSHEY_SIMPLEX

        font_scale = 0.55
        label_thickness = 2

        (text_width, text_height), baseline = (
            cv2.getTextSize(
                label,
                font,
                font_scale,
                label_thickness,
            )
        )

        label_y1 = max(
            0,
            y1 - text_height - baseline - 5,
        )

        label_y2 = y1

        label_x2 = x1 + text_width + 8

        cv2.rectangle(
            image_array,
            (x1, label_y1),
            (label_x2, label_y2),
            (255, 0, 0),
            -1,
        )

        # ----------------------------------------------------
        # Label text
        # ----------------------------------------------------

        cv2.putText(
            image_array,
            label,
            (x1 + 4, y1 - 5),
            font,
            font_scale,
            (255, 255, 255),
            label_thickness,
            cv2.LINE_AA,
        )

    return image_array


# ============================================================
# SUMMARY
# ============================================================

def summarize_detections(
    detections: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Generate a simple summary of detections.
    """

    total = len(detections)

    if total == 0:
        return {
            "total_detections": 0,
            "average_confidence": 0.0,
            "highest_confidence": 0.0,
            "classes": {},
        }

    confidences = [
        float(
            detection.get(
                "confidence",
                0.0,
            )
        )
        for detection in detections
    ]

    class_counts: Dict[str, int] = {}

    for detection in detections:

        class_name = str(
            detection.get(
                "class_name",
                "Unknown",
            )
        )

        class_counts[class_name] = (
            class_counts.get(
                class_name,
                0,
            )
            + 1
        )

    return {
        "total_detections": total,

        "average_confidence": (
            sum(confidences) / len(confidences)
        ),

        "highest_confidence": max(
            confidences
        ),

        "classes": class_counts,
    }


# ============================================================
# MODEL INFORMATION
# ============================================================

def get_model_info(
    model_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Return information about the RoadGuard model.
    """

    if model_path is None:
        model_file = MODEL_PATH
    else:
        model_file = Path(model_path)

        if not model_file.is_absolute():
            model_file = BASE_DIR / model_file

    model_file = model_file.resolve()

    return {
        "model_name": "YOLOv8n",
        "dataset": "RDD2022",
        "model_path": str(model_file),
        "model_exists": model_file.exists(),
        "classes": CLASS_NAMES.copy(),
        "number_of_classes": len(CLASS_NAMES),
    }


# ============================================================
# TEST MODEL
# ============================================================

def test_model(
    model_path: Optional[Union[str, Path]] = None,
) -> bool:
    """
    Test whether the RoadGuard YOLO model can be loaded.

    Returns
    -------
    bool
        True if the model loads successfully.
    """

    try:
        get_model(model_path)
        return True

    except Exception as exc:
        print(
            "\n"
            "RoadGuard model test failed.\n"
            f"{type(exc).__name__}: {exc}\n"
        )

        return False


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("ROADGUARD INFRASTRUCTURE INTELLIGENCE")
    print("Inference Engine")
    print("=" * 60)

    print()
    print("Project directory:")
    print(BASE_DIR)

    print()
    print("Model path:")
    print(MODEL_PATH)

    print()
    print("Model exists:")
    print(MODEL_PATH.exists())

    print()
    print("Classes:")
    for class_id, class_name in CLASS_NAMES.items():
        print(
            f"  {class_id}: {class_name}"
        )

    print()

    if test_model():
        print("Model status: READY")
    else:
        print("Model status: FAILED")

    print("=" * 60)
