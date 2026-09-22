from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

CLASS_NAMES = {
    0: "Longitudinal Crack",
    1: "Transverse Crack",
    2: "Alligator Crack",
    3: "Other Corruption",
    4: "Pothole",
}

CLASS_COLORS = {
    0: (255, 170, 0),
    1: (0, 200, 255),
    2: (255, 80, 180),
    3: (160, 120, 255),
    4: (0, 80, 255),
}

DEFAULT_CONFIDENCE = 0.25
DEFAULT_IOU = 0.45
DEFAULT_IMAGE_SIZE = 640

SEVERITY_WEIGHTS = {
    "Longitudinal Crack": 1.0,
    "Transverse Crack": 1.0,
    "Alligator Crack": 2.0,
    "Other Corruption": 1.25,
    "Pothole": 2.5,
}


def get_model(model_path: Union[str, Path] = "roadguard_best.pt") -> YOLO:
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found: {path.resolve()}\n"
            "Place roadguard_best.pt beside app.py."
        )
    try:
        return YOLO(str(path))
    except Exception as exc:
        raise RuntimeError(f"Could not load model '{path}': {exc}") from exc


def get_model_info(model: YOLO) -> Dict[str, Any]:
    names = getattr(model, "names", None)
    if names is None:
        names = CLASS_NAMES.copy()
    if isinstance(names, list):
        names = {i: str(v) for i, v in enumerate(names)}
    else:
        names = {int(k): str(v) for k, v in dict(names).items()}
    return {
        "model_type": type(model).__name__,
        "task": getattr(model, "task", "detect"),
        "classes": names,
        "num_classes": len(names),
    }


def _to_numpy_image(image: Any) -> np.ndarray:
    if isinstance(image, Image.Image):
        return np.array(image.convert("RGB"))
    if isinstance(image, (str, Path)):
        return np.array(Image.open(image).convert("RGB"))
    if isinstance(image, np.ndarray):
        arr = image
        if arr.ndim == 2:
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
        if arr.ndim != 3 or arr.shape[2] not in (3, 4):
            raise ValueError("Input NumPy image must have shape HxWx3 or HxWx4.")
        if arr.shape[2] == 4:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return arr
    raise TypeError("Use a PIL Image, NumPy array, or image path.")


def _resolve_class_name(model: YOLO, class_id: int) -> str:
    names = getattr(model, "names", None)
    if names is not None:
        try:
            raw = str(names[class_id] if isinstance(names, list) else names.get(class_id, ""))
            aliases = {
                "d00": "Longitudinal Crack",
                "d01": "Transverse Crack",
                "d10": "Alligator Crack",
                "d11": "Other Corruption",
                "d20": "Pothole",
            }
            if raw.strip().lower() in aliases:
                return aliases[raw.strip().lower()]
            if raw in CLASS_NAMES.values():
                return raw
            if raw:
                return raw
        except Exception:
            pass
    return CLASS_NAMES.get(class_id, f"Class {class_id}")


def predict_image(
    model: YOLO,
    image: Any,
    confidence: float = DEFAULT_CONFIDENCE,
    iou: float = DEFAULT_IOU,
    image_size: int = DEFAULT_IMAGE_SIZE,
    device: Optional[Union[str, int]] = None,
    max_det: int = 300,
) -> Dict[str, Any]:
    rgb = _to_numpy_image(image)
    kwargs = {
        "source": rgb,
        "conf": float(confidence),
        "iou": float(iou),
        "imgsz": int(image_size),
        "max_det": int(max_det),
        "verbose": False,
    }
    if device is not None:
        kwargs["device"] = device
    try:
        results = model.predict(**kwargs)
    except Exception as exc:
        raise RuntimeError(f"YOLO inference failed: {exc}") from exc

    detections: List[Dict[str, Any]] = []
    result = results[0] if results else None
    boxes = getattr(result, "boxes", None) if result is not None else None

    if boxes is not None:
        xyxy = boxes.xyxy.detach().cpu().numpy() if boxes.xyxy is not None else np.empty((0, 4))
        confs = boxes.conf.detach().cpu().numpy() if boxes.conf is not None else np.empty((0,))
        classes = boxes.cls.detach().cpu().numpy().astype(int) if boxes.cls is not None else np.empty((0,), dtype=int)
        h, w = rgb.shape[:2]
        for idx, (box, score, class_id) in enumerate(zip(xyxy, confs, classes), 1):
            x1, y1, x2, y2 = map(float, box)
            x1, x2 = sorted((max(0.0, min(x1, w - 1)), max(0.0, min(x2, w - 1))))
            y1, y2 = sorted((max(0.0, min(y1, h - 1)), max(0.0, min(y2, h - 1))))
            bw, bh = x2 - x1, y2 - y1
            detections.append({
                "id": idx,
                "class_id": int(class_id),
                "class_name": _resolve_class_name(model, int(class_id)),
                "confidence": float(score),
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "width": bw, "height": bh, "area": bw * bh,
            })

    return {
        "image": rgb,
        "detections": detections,
        "annotated_image": annotate_image(rgb, detections),
        "raw_result": result,
    }


def annotate_image(image: Any, detections: List[Dict[str, Any]], show_confidence: bool = True, line_thickness: int = 2) -> np.ndarray:
    output = _to_numpy_image(image).copy()
    for det in detections:
        cid = int(det.get("class_id", -1))
        name = str(det.get("class_name", CLASS_NAMES.get(cid, "Unknown")))
        conf = float(det.get("confidence", 0.0))
        x1, y1, x2, y2 = [int(round(det[k])) for k in ("x1", "y1", "x2", "y2")]
        rgb_color = CLASS_COLORS.get(cid, (0, 255, 0))
        bgr = tuple(reversed(rgb_color))
        cv2.rectangle(output, (x1, y1), (x2, y2), bgr, max(1, int(line_thickness)))
        label = f"{name} {conf:.0%}" if show_confidence else name
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ly = max(th + 8, y1)
        cv2.rectangle(output, (max(0, x1), ly - th - 8), (max(0, x1) + tw + 8, ly), bgr, -1)
        cv2.putText(output, label, (max(0, x1) + 4, ly - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return output


def summarize_detections(detections: List[Dict[str, Any]]) -> Dict[str, Any]:
    counts = {name: 0 for name in CLASS_NAMES.values()}
    conf_sums = {name: 0.0 for name in CLASS_NAMES.values()}
    for det in detections:
        name = str(det.get("class_name", "Unknown"))
        counts.setdefault(name, 0)
        conf_sums.setdefault(name, 0.0)
        counts[name] += 1
        conf_sums[name] += float(det.get("confidence", 0.0))
    details = {
        name: {"count": count, "average_confidence": conf_sums[name] / count if count else 0.0}
        for name, count in counts.items()
    }
    total = len(detections)
    return {
        "total_detections": total,
        "counts": counts,
        "average_confidence": sum(float(d.get("confidence", 0)) for d in detections) / total if total else 0.0,
        "class_details": details,
    }


def calculate_severity(detections: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not detections:
        return {
            "severity": "No Detected Damage",
            "risk": "Low",
            "priority": "Routine Monitoring",
            "score": 0.0,
            "weighted_score": 0.0,
            "explanation": "No objects were detected above the selected confidence threshold.",
        }
    weighted = 0.0
    score = 0.0
    for det in detections:
        name = str(det.get("class_name", "Unknown"))
        weight = SEVERITY_WEIGHTS.get(name, 1.0)
        conf = float(det.get("confidence", 0.0))
        weighted += weight
        score += weight * conf
    count = len(detections)
    if score >= 8.0 or count >= 8:
        severity, risk, priority = "Critical", "High", "Immediate Inspection"
    elif score >= 5.0 or count >= 5:
        severity, risk, priority = "High", "High", "Priority Maintenance"
    elif score >= 2.5 or count >= 3:
        severity, risk, priority = "Moderate", "Medium", "Scheduled Maintenance"
    else:
        severity, risk, priority = "Low", "Low", "Routine Monitoring"
    return {
        "severity": severity,
        "risk": risk,
        "priority": priority,
        "score": round(score, 2),
        "weighted_score": round(weighted, 2),
        "explanation": "Severity is an application heuristic based on detected damage categories and confidence; it is not a substitute for professional road inspection.",
    }


def get_severity(detections: List[Dict[str, Any]]) -> Dict[str, Any]:
    return calculate_severity(detections)


def detections_to_rows(detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [{
        "ID": int(d.get("id", 0)),
        "Damage Type": str(d.get("class_name", "Unknown")),
        "Confidence": round(float(d.get("confidence", 0)), 4),
        "X1": round(float(d.get("x1", 0)), 1),
        "Y1": round(float(d.get("y1", 0)), 1),
        "X2": round(float(d.get("x2", 0)), 1),
        "Y2": round(float(d.get("y2", 0)), 1),
        "Area": round(float(d.get("area", 0)), 1),
    } for d in detections]
