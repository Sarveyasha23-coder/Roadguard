"""
RoadGuard Infrastructure Intelligence
======================================

Severity & Risk Decision Engine

This module converts computer-vision detections into
interpretable road-damage severity and priority information.

INPUT
-----
A YOLO detection containing:

    - damage type
    - confidence
    - bounding box
    - image dimensions

OUTPUT
------
    - damage severity
    - damage area
    - damage risk score
    - priority
    - recommendation
    - overall road risk
    - overall road priority

IMPORTANT
---------
This is an ENGINEERED decision-support system.

The RDD2022 dataset provides road-damage annotations,
but it does NOT directly provide real-world repair urgency
or accident-risk labels.

Therefore, RoadGuard's severity and risk values are
transparent heuristic estimates, not probabilities of
an accident or guaranteed maintenance requirements.

Author:
    RoadGuard Team
"""


from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


# ============================================================
# VERSION
# ============================================================

SEVERITY_ENGINE_VERSION = "1.0.0"


# ============================================================
# ROAD DAMAGE CLASSES
# ============================================================

CLASS_NAMES: Dict[int, str] = {
    0: "Longitudinal Crack",
    1: "Transverse Crack",
    2: "Alligator Crack",
    3: "Other Corruption",
    4: "Pothole",
}


# ============================================================
# DAMAGE WEIGHTS
# ============================================================
#
# These are engineering assumptions used by RoadGuard.
#
# They are NOT learned from another dataset.
#
# Higher value = greater contribution to the risk score.
#
# Potholes receive a higher value because they represent
# a different type of physical road defect than cracks.
#
# These values should be described in your project report
# as "engineered class weights".
# ============================================================

DAMAGE_WEIGHTS: Dict[str, float] = {

    "Longitudinal Crack": 0.70,

    "Transverse Crack": 0.65,

    "Alligator Crack": 0.85,

    "Other Corruption": 0.50,

    "Pothole": 1.00,
}


# ============================================================
# SCORING CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class SeverityConfig:
    """
    Configuration for RoadGuard's severity engine.

    The weights must add up to 1.0 for each scoring system.
    """

    # --------------------------------------------------------
    # Risk score weights
    # --------------------------------------------------------

    risk_damage_weight: float = 0.45

    risk_confidence_weight: float = 0.30

    risk_area_weight: float = 0.25

    # --------------------------------------------------------
    # Severity score weights
    # --------------------------------------------------------

    severity_area_weight: float = 0.70

    severity_confidence_weight: float = 0.30

    # --------------------------------------------------------
    # Risk thresholds
    # --------------------------------------------------------

    high_risk_threshold: float = 70.0

    medium_risk_threshold: float = 40.0

    # --------------------------------------------------------
    # Severity thresholds
    # --------------------------------------------------------

    high_severity_threshold: float = 60.0

    medium_severity_threshold: float = 30.0

    # --------------------------------------------------------
    # Minimum confidence used by the decision engine
    # --------------------------------------------------------

    minimum_confidence: float = 0.0

    # --------------------------------------------------------
    # Area normalization
    #
    # A bounding box covering approximately 10% of the
    # image is treated as a strong visible-area signal.
    #
    # This does NOT mean 10% physical road damage.
    # It only refers to the detected bounding-box area
    # relative to the image.
    # --------------------------------------------------------

    area_reference_ratio: float = 0.10


DEFAULT_CONFIG = SeverityConfig()


# ============================================================
# VALIDATION HELPERS
# ============================================================

def _clamp(
    value: float,
    minimum: float = 0.0,
    maximum: float = 100.0,
) -> float:
    """
    Clamp a number to a specified range.
    """

    return max(
        minimum,
        min(
            maximum,
            float(value),
        ),
    )


def _safe_float(
    value: Any,
    default: float = 0.0,
) -> float:
    """
    Safely convert a value to float.
    """

    try:
        result = float(value)

        if math.isnan(result):
            return default

        if math.isinf(result):
            return default

        return result

    except (
        TypeError,
        ValueError,
    ):
        return default


def validate_config(
    config: SeverityConfig,
) -> None:
    """
    Validate scoring configuration.
    """

    risk_total = (
        config.risk_damage_weight
        + config.risk_confidence_weight
        + config.risk_area_weight
    )

    severity_total = (
        config.severity_area_weight
        + config.severity_confidence_weight
    )

    if not math.isclose(
        risk_total,
        1.0,
        abs_tol=1e-6,
    ):
        raise ValueError(
            "Risk weights must sum to 1.0. "
            f"Current total: {risk_total}"
        )

    if not math.isclose(
        severity_total,
        1.0,
        abs_tol=1e-6,
    ):
        raise ValueError(
            "Severity weights must sum to 1.0. "
            f"Current total: {severity_total}"
        )

    if not (
        0.0 <=
        config.high_risk_threshold <=
        100.0
    ):
        raise ValueError(
            "high_risk_threshold must be between 0 and 100."
        )

    if not (
        0.0 <=
        config.medium_risk_threshold <=
        100.0
    ):
        raise ValueError(
            "medium_risk_threshold must be between 0 and 100."
        )

    if (
        config.medium_risk_threshold
        >=
        config.high_risk_threshold
    ):
        raise ValueError(
            "Medium risk threshold must be lower "
            "than high risk threshold."
        )

    if not (
        0.0 <
        config.area_reference_ratio <=
        1.0
    ):
        raise ValueError(
            "area_reference_ratio must be > 0 and <= 1."
        )


# Validate the default configuration when the module loads.
validate_config(DEFAULT_CONFIG)


# ============================================================
# BOUNDING BOX UTILITIES
# ============================================================

def normalize_box(
    box: Sequence[float],
) -> Tuple[float, float, float, float]:
    """
    Normalize a bounding box into:

        x1, y1, x2, y2

    Parameters
    ----------
    box:
        Four numerical values.

    Returns
    -------
    Tuple[float, float, float, float]
    """

    if len(box) != 4:
        raise ValueError(
            "Bounding box must contain exactly "
            "four values: x1, y1, x2, y2."
        )

    x1 = _safe_float(box[0])
    y1 = _safe_float(box[1])
    x2 = _safe_float(box[2])
    y2 = _safe_float(box[3])

    # Make sure coordinates are ordered.
    if x2 < x1:
        x1, x2 = x2, x1

    if y2 < y1:
        y1, y2 = y2, y1

    return (
        x1,
        y1,
        x2,
        y2,
    )


def bounding_box_area(
    box: Sequence[float],
) -> float:
    """
    Calculate bounding-box area in pixels.
    """

    x1, y1, x2, y2 = normalize_box(box)

    width = max(
        0.0,
        x2 - x1,
    )

    height = max(
        0.0,
        y2 - y1,
    )

    return width * height


def bounding_box_center(
    box: Sequence[float],
) -> Tuple[float, float]:
    """
    Calculate center point of a bounding box.
    """

    x1, y1, x2, y2 = normalize_box(box)

    return (
        (x1 + x2) / 2.0,
        (y1 + y2) / 2.0,
    )


def bounding_box_dimensions(
    box: Sequence[float],
) -> Tuple[float, float]:
    """
    Return bounding-box width and height.
    """

    x1, y1, x2, y2 = normalize_box(box)

    return (
        max(0.0, x2 - x1),
        max(0.0, y2 - y1),
    )


# ============================================================
# IMAGE AREA
# ============================================================

def image_area(
    image_width: float,
    image_height: float,
) -> float:
    """
    Calculate image area.
    """

    width = max(
        1.0,
        _safe_float(image_width),
    )

    height = max(
        1.0,
        _safe_float(image_height),
    )

    return width * height


# ============================================================
# DAMAGE AREA RATIO
# ============================================================

def calculate_area_ratio(
    box: Sequence[float],
    image_width: float,
    image_height: float,
) -> float:
    """
    Calculate the detected bounding-box area as a fraction
    of the entire image.

    Example:

        0.05 = 5% of image area
    """

    box_area = bounding_box_area(
        box
    )

    total_area = image_area(
        image_width,
        image_height,
    )

    return _clamp(
        box_area / total_area,
        0.0,
        1.0,
    )


def calculate_area_percentage(
    area_ratio: float,
) -> float:
    """
    Convert area ratio to percentage.

    Example:

        0.05 -> 5.0
    """

    return round(
        _clamp(area_ratio, 0.0, 1.0) * 100.0,
        3,
    )


# ============================================================
# NORMALIZED AREA SCORE
# ============================================================

def calculate_area_score(
    area_ratio: float,
    config: SeverityConfig = DEFAULT_CONFIG,
) -> float:
    """
    Convert bounding-box area ratio into a 0-100 score.

    The area reference ratio controls how quickly the score
    approaches 100.

    Example with reference ratio = 0.10:

        0.01 -> 10
        0.05 -> 50
        0.10 -> 100

    This is an engineered normalization, not a physical
    measurement of actual damage depth or road deterioration.
    """

    area_ratio = _clamp(
        area_ratio,
        0.0,
        1.0,
    )

    reference = max(
        config.area_reference_ratio,
        1e-9,
    )

    score = (
        area_ratio /
        reference
    ) * 100.0

    return round(
        _clamp(score),
        3,
    )


# ============================================================
# CONFIDENCE SCORE
# ============================================================

def calculate_confidence_score(
    confidence: float,
) -> float:
    """
    Convert YOLO confidence from:

        0.0 - 1.0

    to:

        0 - 100
    """

    confidence = _clamp(
        _safe_float(confidence),
        0.0,
        1.0,
    )

    return round(
        confidence * 100.0,
        3,
    )


# ============================================================
# DAMAGE CLASS SCORE
# ============================================================

def get_damage_weight(
    damage_name: str,
) -> float:
    """
    Return engineered importance weight for a damage class.
    """

    return DAMAGE_WEIGHTS.get(
        damage_name,
        0.50,
    )


def get_damage_class_score(
    damage_name: str,
) -> float:
    """
    Convert damage weight into 0-100 score.
    """

    return round(
        get_damage_weight(
            damage_name
        ) * 100.0,
        3,
    )


# ============================================================
# SEVERITY SCORE
# ============================================================

def calculate_severity_score(
    area_ratio: float,
    confidence: float,
    config: SeverityConfig = DEFAULT_CONFIG,
) -> float:
    """
    Calculate a 0-100 engineered severity score.

    Components:

        70% visible-area signal
        30% model-confidence signal

    Note:
        This is not a ground-truth severity measurement.
    """

    area_score = calculate_area_score(
        area_ratio,
        config,
    )

    confidence_score = calculate_confidence_score(
        confidence
    )

    score = (
        config.severity_area_weight
        * area_score
        +
        config.severity_confidence_weight
        * confidence_score
    )

    return round(
        _clamp(score),
        2,
    )


# ============================================================
# SEVERITY LABEL
# ============================================================

def severity_label(
    severity_score: float,
    config: SeverityConfig = DEFAULT_CONFIG,
) -> str:
    """
    Convert severity score to:

        LOW
        MEDIUM
        HIGH
    """

    score = _clamp(
        severity_score
    )

    if (
        score >=
        config.high_severity_threshold
    ):
        return "HIGH"

    if (
        score >=
        config.medium_severity_threshold
    ):
        return "MEDIUM"

    return "LOW"


# ============================================================
# RISK SCORE
# ============================================================

def calculate_risk_score(
    damage_name: str,
    confidence: float,
    area_ratio: float,
    config: SeverityConfig = DEFAULT_CONFIG,
) -> float:
    """
    Calculate RoadGuard's 0-100 engineered risk score.

    Formula:

        Risk =
            45% damage type
          + 30% confidence
          + 25% visible area

    This should NOT be described as:
        "87% chance of an accident"

    Instead describe it as:
        "RoadGuard risk score: 87/100"

    It is a prioritization signal.
    """

    damage_score = get_damage_class_score(
        damage_name
    )

    confidence_score = calculate_confidence_score(
        confidence
    )

    area_score = calculate_area_score(
        area_ratio,
        config,
    )

    risk = (
        config.risk_damage_weight
        * damage_score
        +
        config.risk_confidence_weight
        * confidence_score
        +
        config.risk_area_weight
        * area_score
    )

    return round(
        _clamp(risk),
        2,
    )


# ============================================================
# PRIORITY
# ============================================================

def priority_label(
    risk_score: float,
    config: SeverityConfig = DEFAULT_CONFIG,
) -> str:
    """
    Convert risk score to operational priority.
    """

    score = _clamp(
        risk_score
    )

    if (
        score >=
        config.high_risk_threshold
    ):
        return "HIGH"

    if (
        score >=
        config.medium_risk_threshold
    ):
        return "MEDIUM"

    return "LOW"


# ============================================================
# PRIORITY LEVEL
# ============================================================

def priority_level(
    priority: str,
) -> int:
    """
    Convert priority label to a numerical level.

        HIGH   = 3
        MEDIUM = 2
        LOW    = 1
    """

    mapping = {
        "HIGH": 3,
        "MEDIUM": 2,
        "LOW": 1,
    }

    return mapping.get(
        priority.upper(),
        1,
    )


# ============================================================
# RECOMMENDATION ENGINE
# ============================================================

def generate_recommendation(
    damage_name: str,
    severity: str,
    risk_score: float,
    priority: str,
) -> str:
    """
    Generate an explainable maintenance recommendation.

    These are recommendations for inspection/monitoring,
    NOT guaranteed engineering instructions.
    """

    damage = damage_name.lower()

    if priority == "HIGH":

        if "pothole" in damage:
            return (
                "High-priority pothole detected. "
                "Schedule prompt field inspection "
                "and assess for maintenance."
            )

        if "alligator" in damage:
            return (
                "High-priority alligator cracking detected. "
                "Schedule prompt field inspection "
                "to assess pavement condition."
            )

        if "crack" in damage:
            return (
                "High-priority cracking detected. "
                "Schedule prompt inspection and "
                "evaluate the affected road section."
            )

        return (
            "High-priority road damage detected. "
            "Schedule prompt field inspection."
        )

    if priority == "MEDIUM":

        if "pothole" in damage:
            return (
                "Moderate-priority pothole detected. "
                "Schedule inspection and monitor progression."
            )

        if "crack" in damage:
            return (
                "Moderate-priority cracking detected. "
                "Schedule inspection and monitor progression."
            )

        return (
            "Moderate-priority road damage detected. "
            "Schedule inspection and monitor condition."
        )

    # LOW

    return (
        "Low-priority visual damage detected. "
        "Continue monitoring during future inspections."
    )


# ============================================================
# EXPLANATION ENGINE
# ============================================================

def generate_explanation(
    damage_name: str,
    confidence: float,
    area_ratio: float,
    severity_score: float,
    risk_score: float,
    priority: str,
) -> str:
    """
    Generate a human-readable explanation of the score.

    This is useful for your dashboard and judges because
    it makes the AI decision more interpretable.
    """

    area_percentage = (
        calculate_area_percentage(
            area_ratio
        )
    )

    confidence_percentage = (
        calculate_confidence_score(
            confidence
        )
    )

    return (
        f"RoadGuard detected {damage_name} "
        f"with {confidence_percentage:.1f}% model confidence. "
        f"The detected bounding box covers approximately "
        f"{area_percentage:.2f}% of the image. "
        f"The engineered severity score is "
        f"{severity_score:.1f}/100 and the engineered "
        f"risk score is {risk_score:.1f}/100, "
        f"resulting in {priority} priority."
    )


# ============================================================
# SINGLE DETECTION ANALYSIS
# ============================================================

def analyze_detection(
    damage_name: str,
    confidence: float,
    box: Sequence[float],
    image_width: float,
    image_height: float,
    config: SeverityConfig = DEFAULT_CONFIG,
    detection_id: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Analyze one YOLO detection.

    This is the main function that inference.py can call.
    """

    validate_config(
        config
    )

    confidence = _clamp(
        _safe_float(confidence),
        0.0,
        1.0,
    )

    # --------------------------------------------------------
    # Geometry
    # --------------------------------------------------------

    normalized_box = normalize_box(
        box
    )

    width, height = bounding_box_dimensions(
        normalized_box
    )

    box_area = bounding_box_area(
        normalized_box
    )

    center_x, center_y = bounding_box_center(
        normalized_box
    )

    # --------------------------------------------------------
    # Area
    # --------------------------------------------------------

    area_ratio = calculate_area_ratio(
        normalized_box,
        image_width,
        image_height,
    )

    area_percentage = calculate_area_percentage(
        area_ratio
    )

    area_score = calculate_area_score(
        area_ratio,
        config,
    )

    # --------------------------------------------------------
    # Confidence
    # --------------------------------------------------------

    confidence_score = calculate_confidence_score(
        confidence
    )

    # --------------------------------------------------------
    # Damage class
    # --------------------------------------------------------

    damage_weight = get_damage_weight(
        damage_name
    )

    damage_class_score = get_damage_class_score(
        damage_name
    )

    # --------------------------------------------------------
    # Severity
    # --------------------------------------------------------

    severity_score = calculate_severity_score(
        area_ratio=area_ratio,
        confidence=confidence,
        config=config,
    )

    severity = severity_label(
        severity_score,
        config,
    )

    # --------------------------------------------------------
    # Risk
    # --------------------------------------------------------

    risk_score = calculate_risk_score(
        damage_name=damage_name,
        confidence=confidence,
        area_ratio=area_ratio,
        config=config,
    )

    # --------------------------------------------------------
    # Priority
    # --------------------------------------------------------

    priority = priority_label(
        risk_score,
        config,
    )

    priority_numeric = priority_level(
        priority
    )

    # --------------------------------------------------------
    # Recommendation
    # --------------------------------------------------------

    recommendation = generate_recommendation(
        damage_name=damage_name,
        severity=severity,
        risk_score=risk_score,
        priority=priority,
    )

    # --------------------------------------------------------
    # Explanation
    # --------------------------------------------------------

    explanation = generate_explanation(
        damage_name=damage_name,
        confidence=confidence,
        area_ratio=area_ratio,
        severity_score=severity_score,
        risk_score=risk_score,
        priority=priority,
    )

    # --------------------------------------------------------
    # Final structured result
    # --------------------------------------------------------

    result = {
        "id": detection_id,

        "damage_type": damage_name,

        "confidence": round(
            confidence,
            4,
        ),

        "confidence_percentage": round(
            confidence_score,
            2,
        ),

        "bounding_box": {
            "x1": round(
                normalized_box[0],
                2,
            ),
            "y1": round(
                normalized_box[1],
                2,
            ),
            "x2": round(
                normalized_box[2],
                2,
            ),
            "y2": round(
                normalized_box[3],
                2,
            ),
        },

        "geometry": {
            "width_pixels": round(
                width,
                2,
            ),
            "height_pixels": round(
                height,
                2,
            ),
            "area_pixels": round(
                box_area,
                2,
            ),
            "center_x": round(
                center_x,
                2,
            ),
            "center_y": round(
                center_y,
                2,
            ),
        },

        "image_dimensions": {
            "width": int(
                image_width
            ),
            "height": int(
                image_height
            ),
        },

        "damage_weight": round(
            damage_weight,
            3,
        ),

        "damage_class_score": round(
            damage_class_score,
            2,
        ),

        "area_ratio": round(
            area_ratio,
            5,
        ),

        "area_percentage": round(
            area_percentage,
            3,
        ),

        "area_score": round(
            area_score,
            2,
        ),

        "severity_score": round(
            severity_score,
            2,
        ),

        "severity": severity,

        "risk_score": round(
            risk_score,
            2,
        ),

        "priority": priority,

        "priority_level": priority_numeric,

        "recommendation": recommendation,

        "explanation": explanation,
    }

    return result


# ============================================================
# OVERALL ROAD ANALYSIS
# ============================================================

def analyze_detections(
    detections: Iterable[Dict[str, Any]],
    image_width: int,
    image_height: int,
    config: SeverityConfig = DEFAULT_CONFIG,
) -> Dict[str, Any]:
    """
    Analyze multiple YOLO detections.

    Expected input:

        [
            {
                "damage_type": "Pothole",
                "confidence": 0.91,
                "bounding_box": {
                    "x1": ...,
                    "y1": ...,
                    "x2": ...,
                    "y2": ...
                }
            }
        ]

    Returns a complete RoadGuard road-level analysis.
    """

    validate_config(
        config
    )

    analyzed: List[Dict[str, Any]] = []

    for index, detection in enumerate(
        detections,
        start=1,
    ):

        damage_name = detection.get(
            "damage_type",
            "Unknown",
        )

        confidence = detection.get(
            "confidence",
            0.0,
        )

        box_data = detection.get(
            "bounding_box"
        )

        if box_data is None:
            continue

        if isinstance(
            box_data,
            dict,
        ):

            box = [
                box_data.get("x1", 0),
                box_data.get("y1", 0),
                box_data.get("x2", 0),
                box_data.get("y2", 0),
            ]

        else:

            box = box_data

        result = analyze_detection(
            damage_name=damage_name,
            confidence=confidence,
            box=box,
            image_width=image_width,
            image_height=image_height,
            config=config,
            detection_id=index,
        )

        analyzed.append(
            result
        )

    # ========================================================
    # NO DAMAGE
    # ========================================================

    if not analyzed:

        return {
            "detections_count": 0,
            "damage_types_detected": [],
            "overall_risk_score": 0.0,
            "overall_severity_score": 0.0,
            "overall_severity": "LOW",
            "overall_priority": "LOW",
            "highest_risk_detection": None,
            "highest_severity_detection": None,
            "recommendation": (
                "No road damage was detected. "
                "Continue normal monitoring."
            ),
            "detections": [],
        }

    # ========================================================
    # HIGHEST RISK
    # ========================================================

    highest_risk = max(
        analyzed,
        key=lambda item: item["risk_score"],
    )

    highest_severity = max(
        analyzed,
        key=lambda item: item["severity_score"],
    )

    # ========================================================
    # OVERALL RISK
    # ========================================================
    #
    # We use the highest individual risk as the primary
    # road-level risk because one severe defect can warrant
    # attention even when other defects are minor.
    #
    # We do NOT simply add scores because that could create
    # an artificial score above 100.
    # ========================================================

    overall_risk = highest_risk[
        "risk_score"
    ]

    overall_severity_score = highest_severity[
        "severity_score"
    ]

    overall_priority = priority_label(
        overall_risk,
        config,
    )

    overall_severity = severity_label(
        overall_severity_score,
        config,
    )

    # ========================================================
    # DAMAGE TYPES
    # ========================================================

    damage_types = sorted(
        set(
            item["damage_type"]
            for item in analyzed
        )
    )

    # ========================================================
    # COUNTS
    # ========================================================

    high_count = sum(
        item["priority"] == "HIGH"
        for item in analyzed
    )

    medium_count = sum(
        item["priority"] == "MEDIUM"
        for item in analyzed
    )

    low_count = sum(
        item["priority"] == "LOW"
        for item in analyzed
    )

    # ========================================================
    # RECOMMENDATION
    # ========================================================

    if overall_priority == "HIGH":

        recommendation = (
            "High-priority road damage detected. "
            "The affected section should be considered "
            "for prompt field inspection."
        )

    elif overall_priority == "MEDIUM":

        recommendation = (
            "Moderate-priority road damage detected. "
            "Schedule inspection and monitor the "
            "affected road section."
        )

    else:

        recommendation = (
            "Only low-priority visual damage was detected. "
            "Continue monitoring during future inspections."
        )

    # ========================================================
    # RETURN
    # ========================================================

    return {
        "detections_count": len(
            analyzed
        ),

        "damage_types_detected": damage_types,

        "overall_risk_score": round(
            overall_risk,
            2,
        ),

        "overall_severity_score": round(
            overall_severity_score,
            2,
        ),

        "overall_severity":
            overall_severity,

        "overall_priority":
            overall_priority,

        "priority_counts": {
            "HIGH": high_count,
            "MEDIUM": medium_count,
            "LOW": low_count,
        },

        "highest_risk_detection":
            highest_risk,

        "highest_severity_detection":
            highest_severity,

        "recommendation":
            recommendation,

        "detections":
            analyzed,
    }


# ============================================================
# SIMPLE DETECTION ADAPTER
# ============================================================

def analyze_yolo_detection(
    damage_class_id: int,
    confidence: float,
    xyxy: Sequence[float],
    image_width: int,
    image_height: int,
    config: SeverityConfig = DEFAULT_CONFIG,
    detection_id: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Convenience function for directly connecting this module
    to a YOLO result.

    Example:

        analyze_yolo_detection(
            damage_class_id=4,
            confidence=0.91,
            xyxy=[100, 200, 500, 600],
            image_width=1920,
            image_height=1080,
        )
    """

    damage_name = CLASS_NAMES.get(
        int(damage_class_id),
        f"Unknown Class {damage_class_id}",
    )

    return analyze_detection(
        damage_name=damage_name,
        confidence=confidence,
        box=xyxy,
        image_width=image_width,
        image_height=image_height,
        config=config,
        detection_id=detection_id,
    )


# ============================================================
# SCORE BREAKDOWN
# ============================================================

def get_score_breakdown(
    damage_name: str,
    confidence: float,
    area_ratio: float,
    config: SeverityConfig = DEFAULT_CONFIG,
) -> Dict[str, Any]:
    """
    Return the individual components contributing to
    the RoadGuard risk score.

    This is particularly useful for explainable AI dashboards.
    """

    damage_score = get_damage_class_score(
        damage_name
    )

    confidence_score = calculate_confidence_score(
        confidence
    )

    area_score = calculate_area_score(
        area_ratio,
        config,
    )

    damage_contribution = (
        config.risk_damage_weight
        * damage_score
    )

    confidence_contribution = (
        config.risk_confidence_weight
        * confidence_score
    )

    area_contribution = (
        config.risk_area_weight
        * area_score
    )

    final_score = (
        damage_contribution
        +
        confidence_contribution
        +
        area_contribution
    )

    return {
        "damage_type": damage_name,

        "damage_score": round(
            damage_score,
            2,
        ),

        "confidence_score": round(
            confidence_score,
            2,
        ),

        "area_score": round(
            area_score,
            2,
        ),

        "weights": {
            "damage": config.risk_damage_weight,
            "confidence":
                config.risk_confidence_weight,
            "area":
                config.risk_area_weight,
        },

        "contributions": {
            "damage": round(
                damage_contribution,
                2,
            ),
            "confidence": round(
                confidence_contribution,
                2,
            ),
            "area": round(
                area_contribution,
                2,
            ),
        },

        "final_risk_score": round(
            _clamp(final_score),
            2,
        ),
    }


# ============================================================
# BATCH ANALYSIS
# ============================================================

def analyze_batch(
    roads: Iterable[Dict[str, Any]],
    config: SeverityConfig = DEFAULT_CONFIG,
) -> List[Dict[str, Any]]:
    """
    Analyze multiple road/image records.

    Each record should contain:

        {
            "image": "road1.jpg",
            "image_width": 1920,
            "image_height": 1080,
            "detections": [...]
        }
    """

    results = []

    for road in roads:

        image_name = road.get(
            "image",
            "unknown",
        )

        width = int(
            road.get(
                "image_width",
                1,
            )
        )

        height = int(
            road.get(
                "image_height",
                1,
            )
        )

        detections = road.get(
            "detections",
            [],
        )

        analysis = analyze_detections(
            detections=detections,
            image_width=width,
            image_height=height,
            config=config,
        )

        results.append({
            "image": image_name,
            "analysis": analysis,
        })

    return results


# ============================================================
# SAVE ANALYSIS
# ============================================================

def save_analysis_json(
    analysis: Dict[str, Any],
    output_path: str | Path,
) -> None:
    """
    Save a RoadGuard analysis as JSON.
    """

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            analysis,
            file,
            indent=4,
            ensure_ascii=False,
        )


# ============================================================
# LOAD ANALYSIS
# ============================================================

def load_analysis_json(
    input_path: str | Path,
) -> Dict[str, Any]:
    """
    Load a previously generated RoadGuard JSON report.
    """

    input_path = Path(
        input_path
    )

    if not input_path.exists():
        raise FileNotFoundError(
            f"Analysis file not found: "
            f"{input_path}"
        )

    with open(
        input_path,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(
            file
        )


# ============================================================
# HUMAN-READABLE SUMMARY
# ============================================================

def format_summary(
    analysis: Dict[str, Any],
) -> str:
    """
    Create a clean text summary for terminal output
    or application interfaces.
    """

    lines = []

    lines.append(
        "=" * 60
    )

    lines.append(
        "ROADGUARD INFRASTRUCTURE ANALYSIS"
    )

    lines.append(
        "=" * 60
    )

    lines.append(
        f"Damage detected : "
        f"{analysis.get('detections_count', 0)}"
    )

    lines.append(
        f"Overall risk    : "
        f"{analysis.get('overall_risk_score', 0):.2f}/100"
    )

    lines.append(
        f"Overall severity: "
        f"{analysis.get('overall_severity', 'LOW')}"
    )

    lines.append(
        f"Overall priority: "
        f"{analysis.get('overall_priority', 'LOW')}"
    )

    lines.append(
        ""
    )

    priority_counts = analysis.get(
        "priority_counts",
        {},
    )

    lines.append(
        "Priority distribution:"
    )

    lines.append(
        f"  HIGH   : "
        f"{priority_counts.get('HIGH', 0)}"
    )

    lines.append(
        f"  MEDIUM : "
        f"{priority_counts.get('MEDIUM', 0)}"
    )

    lines.append(
        f"  LOW    : "
        f"{priority_counts.get('LOW', 0)}"
    )

    lines.append(
        ""
    )

    lines.append(
        "Recommendation:"
    )

    lines.append(
        analysis.get(
            "recommendation",
            "",
        )
    )

    lines.append(
        ""
    )

    lines.append(
        "-" * 60
    )

    for detection in analysis.get(
        "detections",
        [],
    ):

        lines.append(
            f"Detection #{detection.get('id')}"
        )

        lines.append(
            f"  Damage      : "
            f"{detection.get('damage_type')}"
        )

        lines.append(
            f"  Confidence  : "
            f"{detection.get('confidence_percentage', 0):.1f}%"
        )

        lines.append(
            f"  Area        : "
            f"{detection.get('area_percentage', 0):.2f}%"
        )

        lines.append(
            f"  Severity    : "
            f"{detection.get('severity')}"
        )

        lines.append(
            f"  Risk        : "
            f"{detection.get('risk_score', 0):.2f}/100"
        )

        lines.append(
            f"  Priority    : "
            f"{detection.get('priority')}"
        )

        lines.append(
            ""
        )

    lines.append(
        "=" * 60
    )

    return "\n".join(
        lines
    )


# ============================================================
# ENGINE INFORMATION
# ============================================================

def get_engine_info() -> Dict[str, Any]:
    """
    Return metadata about the RoadGuard severity engine.
    """

    return {
        "name":
            "RoadGuard Severity & Risk Engine",

        "version":
            SEVERITY_ENGINE_VERSION,

        "classes":
            CLASS_NAMES,

        "damage_weights":
            DAMAGE_WEIGHTS,

        "risk_formula": {
            "damage_type": 0.45,
            "confidence": 0.30,
            "visible_area": 0.25,
        },

        "severity_formula": {
            "visible_area": 0.70,
            "confidence": 0.30,
        },

        "risk_thresholds": {
            "HIGH":
                DEFAULT_CONFIG.high_risk_threshold,
            "MEDIUM":
                DEFAULT_CONFIG.medium_risk_threshold,
        },

        "severity_thresholds": {
            "HIGH":
                DEFAULT_CONFIG.high_severity_threshold,
            "MEDIUM":
                DEFAULT_CONFIG.medium_severity_threshold,
        },

        "note": (
            "Scores are engineered decision-support "
            "signals and are not calibrated probabilities "
            "of accidents or maintenance outcomes."
        ),
    }


# ============================================================
# DEMO
# ============================================================

def demo():
    """
    Run a small demonstration without requiring YOLO.

    Useful for testing severity.py independently.
    """

    print(
        "\n"
        + "=" * 60
    )

    print(
        "ROADGUARD SEVERITY ENGINE DEMO"
    )

    print(
        "=" * 60
    )

    # Example detections generated by the CV model.
    detections = [

        {
            "damage_type":
                "Pothole",

            "confidence":
                0.91,

            "bounding_box": {
                "x1": 400,
                "y1": 300,
                "x2": 850,
                "y2": 650,
            },
        },

        {
            "damage_type":
                "Longitudinal Crack",

            "confidence":
                0.82,

            "bounding_box": {
                "x1": 100,
                "y1": 500,
                "x2": 600,
                "y2": 560,
            },
        },
    ]

    analysis = analyze_detections(
        detections=detections,
        image_width=1920,
        image_height=1080,
    )

    print(
        format_summary(
            analysis
        )
    )

    print(
        "\nJSON representation:"
    )

    print(
        json.dumps(
            analysis,
            indent=4,
        )
    )


# ============================================================
# COMMAND LINE TEST
# ============================================================

def main():
    """
    Run the standalone severity engine demo.

    Usage:

        python severity.py

    """

    demo()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
