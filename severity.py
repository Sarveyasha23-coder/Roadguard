```python
from typing import Any, Dict, List


# ============================================================
# DAMAGE WEIGHTS
# ============================================================

DAMAGE_WEIGHTS = {
    "Longitudinal Crack": 1.00,
    "Transverse Crack": 1.05,
    "Alligator Crack": 1.35,
    "Other Corruption": 1.15,
    "Pothole": 1.40,
}


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def clamp(
    value: float,
    minimum: float,
    maximum: float,
) -> float:

    return max(
        minimum,
        min(value, maximum),
    )


def get_severity_level(
    score: float,
) -> str:

    if score < 30:
        return "Low"

    if score < 60:
        return "Moderate"

    if score < 80:
        return "High"

    return "Critical"


def get_risk_level(
    score: float,
) -> str:

    if score < 25:
        return "Low Risk"

    if score < 50:
        return "Moderate Risk"

    if score < 75:
        return "High Risk"

    return "Critical Risk"


def get_maintenance_priority(
    risk_level: str,
) -> str:

    priorities = {
        "Low Risk": "Routine Monitoring",
        "Moderate Risk": "Scheduled Maintenance",
        "High Risk": "High Priority",
        "Critical Risk": "Immediate Attention",
    }

    return priorities.get(
        risk_level,
        "Scheduled Maintenance",
    )


# ============================================================
# MAINTENANCE RECOMMENDATIONS
# ============================================================

def get_recommendation(
    damage_type: str,
    severity: str,
    risk_level: str,
) -> str:

    if risk_level == "Critical Risk":

        return (
            "Immediate inspection and corrective "
            "maintenance recommended."
        )

    if damage_type == "Pothole":

        if severity in (
            "High",
            "Critical",
        ):

            return (
                "Repair pothole promptly and inspect "
                "surrounding pavement for deterioration."
            )

        return (
            "Schedule pothole repair and monitor "
            "for expansion."
        )

    if damage_type == "Alligator Crack":

        if severity in (
            "High",
            "Critical",
        ):

            return (
                "Conduct detailed pavement inspection "
                "and consider structural rehabilitation."
            )

        return (
            "Seal affected area and monitor "
            "pavement condition."
        )

    if damage_type in (
        "Longitudinal Crack",
        "Transverse Crack",
    ):

        if severity in (
            "High",
            "Critical",
        ):

            return (
                "Inspect crack propagation and perform "
                "appropriate crack sealing or repair."
            )

        return (
            "Monitor crack development and consider "
            "preventive crack sealing."
        )

    if damage_type == "Other Corruption":

        return (
            "Perform site inspection to identify the "
            "cause and determine suitable maintenance."
        )

    return (
        "Inspect the affected area and determine "
        "appropriate maintenance."
    )


# ============================================================
# INDIVIDUAL DETECTION ANALYSIS
# ============================================================

def analyze_detection(
    detection: Dict[str, Any],
) -> Dict[str, Any]:

    damage_type = detection.get(
        "damage_type",
        "Unknown",
    )

    confidence = float(
        detection.get(
            "confidence",
            0.0,
        )
    )

    area_percentage = float(
        detection.get(
            "area_percentage",
            0.0,
        )
    )

    # Confidence contribution
    confidence_score = (
        confidence * 35
    )

    # Area contribution
    area_score = clamp(
        area_percentage * 3.0,
        0,
        35,
    )

    # Damage contribution
    damage_weight = DAMAGE_WEIGHTS.get(
        damage_type,
        1.0,
    )

    damage_score = (
        damage_weight * 20
    )

    # Overall severity
    severity_score = (
        confidence_score
        + area_score
        + damage_score
    )

    severity_score = clamp(
        severity_score,
        0,
        100,
    )

    severity = get_severity_level(
        severity_score
    )

    # Infrastructure risk
    risk_score = (
        severity_score * 0.75
        + damage_score * 1.25
    )

    risk_score = clamp(
        risk_score,
        0,
        100,
    )

    risk_level = get_risk_level(
        risk_score
    )

    priority = get_maintenance_priority(
        risk_level
    )

    recommendation = get_recommendation(
        damage_type,
        severity,
        risk_level,
    )

    return {
        **detection,

        "severity_score": round(
            severity_score,
            2,
        ),

        "severity": severity,

        "risk_score": round(
            risk_score,
            2,
        ),

        "risk_level": risk_level,

        "maintenance_priority": priority,

        "recommended_action": recommendation,
    }


# ============================================================
# COMPLETE IMAGE ANALYSIS
# ============================================================

def analyze_severity(
    detections: List[
        Dict[str, Any]
    ],
) -> Dict[str, Any]:

    analyzed_detections = [
        analyze_detection(
            detection
        )
        for detection in detections
    ]

    # No detections
    if not analyzed_detections:

        return {
            "detections": [],

            "overall_severity": (
                "No Damage"
            ),

            "overall_risk": (
                "Low Risk"
            ),

            "overall_risk_score": 0.0,

            "maintenance_priority": (
                "No Action Required"
            ),

            "overall_recommendation": (
                "No visible road damage "
                "was detected."
            ),
        }

    # Find highest-risk detection
    highest_risk_detection = max(
        analyzed_detections,
        key=lambda item: item[
            "risk_score"
        ],
    )

    overall_risk_score = (
        highest_risk_detection[
            "risk_score"
        ]
    )

    overall_risk = (
        highest_risk_detection[
            "risk_level"
        ]
    )

    overall_severity = (
        highest_risk_detection[
            "severity"
        ]
    )

    # Multiple detections increase
    # overall risk slightly.
    detection_count = len(
        analyzed_detections
    )

    if detection_count >= 5:

        overall_risk_score = clamp(
            overall_risk_score + 10,
            0,
            100,
        )

    elif detection_count >= 3:

        overall_risk_score = clamp(
            overall_risk_score + 5,
            0,
            100,
        )

    overall_risk = get_risk_level(
        overall_risk_score
    )

    maintenance_priority = (
        get_maintenance_priority(
            overall_risk
        )
    )

    # Overall recommendation
    if overall_risk == "Critical Risk":

        recommendation = (
            "Immediate road inspection and "
            "corrective maintenance are recommended."
        )

    elif overall_risk == "High Risk":

        recommendation = (
            "Schedule high-priority maintenance "
            "and conduct a detailed pavement inspection."
        )

    elif overall_risk == "Moderate Risk":

        recommendation = (
            "Schedule maintenance and continue "
            "monitoring the affected road section."
        )

    else:

        recommendation = (
            "Continue routine monitoring and "
            "preventive maintenance."
        )

    return {
        "detections": analyzed_detections,

        "overall_severity": (
            overall_severity
        ),

        "overall_risk": (
            overall_risk
        ),

        "overall_risk_score": round(
            overall_risk_score,
            2,
        ),

        "maintenance_priority": (
            maintenance_priority
        ),

        "overall_recommendation": (
            recommendation
        ),
    }
```
