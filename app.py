"""
============================================================
ROADGUARD INFRASTRUCTURE INTELLIGENCE
============================================================

Computer Vision Based Road Damage Detection & Prioritization

Main Components
---------------

1. YOLOv8n
   Detects road damage.

2. inference.py
   Handles computer-vision inference.

3. severity.py
   Converts detections into:
       - Severity
       - Risk
       - Priority
       - Recommendation

4. app.py
   Streamlit dashboard.

Dataset
-------
RDD2022 is the single dataset used for the project.

This application does NOT require another dataset.

Model
-----
Place your trained model here:

    roadguard_best.pt

Project structure:

    RoadGuard/
    |
    |-- app.py
    |-- inference.py
    |-- severity.py
    |-- roadguard_best.pt
    |-- requirements.txt
    |-- README.md

Run
---
    streamlit run app.py

============================================================
"""


# ============================================================
# IMPORTS
# ============================================================

from __future__ import annotations

import io
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="RoadGuard Infrastructure Intelligence",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "roadguard_best.pt"

INFERENCE_FILE = BASE_DIR / "inference.py"

SEVERITY_FILE = BASE_DIR / "severity.py"


# ============================================================
# APPLICATION CONSTANTS
# ============================================================

APP_NAME = "RoadGuard Infrastructure Intelligence"

APP_VERSION = "1.0.0"

MODEL_NAME = "YOLOv8n"

DATASET_NAME = "RDD2022"

SUPPORTED_IMAGE_TYPES = [
    "jpg",
    "jpeg",
    "png",
    "webp",
]


# ============================================================
# CLASS NAMES
# ============================================================

CLASS_NAMES = {
    0: "Longitudinal Crack",
    1: "Transverse Crack",
    2: "Alligator Crack",
    3: "Other Corruption",
    4: "Pothole",
}


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ---------------------------------------------------- */
    /* GLOBAL */
    /* ---------------------------------------------------- */

    .main {
        padding-top: 1rem;
    }

    /* ---------------------------------------------------- */
    /* HEADER */
    /* ---------------------------------------------------- */

    .roadguard-header {
        padding: 1.5rem;
        border-radius: 18px;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(128,128,128,0.25);
        background: linear-gradient(
            135deg,
            rgba(30,30,30,0.95),
            rgba(55,55,55,0.95)
        );
    }

    .roadguard-title {
        font-size: 2.4rem;
        font-weight: 800;
        margin-bottom: 0.25rem;
    }

    .roadguard-subtitle {
        font-size: 1.05rem;
        opacity: 0.85;
    }

    /* ---------------------------------------------------- */
    /* CARDS */
    /* ---------------------------------------------------- */

    .info-card {
        padding: 1rem;
        border-radius: 14px;
        border: 1px solid rgba(128,128,128,0.25);
        margin-bottom: 1rem;
    }

    .metric-card {
        padding: 1rem;
        border-radius: 14px;
        border: 1px solid rgba(128,128,128,0.25);
        text-align: center;
        min-height: 110px;
    }

    .metric-title {
        font-size: 0.85rem;
        opacity: 0.75;
    }

    .metric-value {
        font-size: 1.8rem;
        font-weight: 800;
    }

    /* ---------------------------------------------------- */
    /* STATUS */
    /* ---------------------------------------------------- */

    .status-high {
        padding: 0.7rem;
        border-radius: 10px;
        font-weight: 700;
        text-align: center;
    }

    .status-medium {
        padding: 0.7rem;
        border-radius: 10px;
        font-weight: 700;
        text-align: center;
    }

    .status-low {
        padding: 0.7rem;
        border-radius: 10px;
        font-weight: 700;
        text-align: center;
    }

    /* ---------------------------------------------------- */
    /* FOOTER */
    /* ---------------------------------------------------- */

    .roadguard-footer {
        text-align: center;
        padding: 2rem;
        opacity: 0.65;
        font-size: 0.85rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    f"""
    <div class="roadguard-header">

        <div class="roadguard-title">
            🛣️ {APP_NAME}
        </div>

        <div class="roadguard-subtitle">
            Computer Vision for Road Damage Detection,
            Severity Assessment and Maintenance Prioritization
        </div>

    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def clamp(
    value: float,
    minimum: float = 0.0,
    maximum: float = 100.0,
) -> float:

    return max(
        minimum,
        min(
            maximum,
            float(value),
        ),
    )


def safe_float(
    value: Any,
    default: float = 0.0,
) -> float:

    try:

        result = float(value)

        if result != result:
            return default

        return result

    except Exception:

        return default


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource(show_spinner=False)
def load_model(
    model_path: str,
):
    """
    Load YOLO model once and keep it in Streamlit cache.

    This prevents the model from being loaded every time
    the application reruns.
    """

    try:

        from ultralytics import YOLO

    except ImportError as exc:

        raise ImportError(
            "Ultralytics is not installed. "
            "Run: pip install ultralytics"
        ) from exc

    if not Path(model_path).exists():

        raise FileNotFoundError(
            f"Model not found:\n{model_path}\n\n"
            "Place roadguard_best.pt in the same "
            "folder as app.py."
        )

    model = YOLO(
        model_path
    )

    return model


# ============================================================
# YOLO DIRECT INFERENCE
# ============================================================

def run_yolo_inference(
    model,
    image: Image.Image,
    confidence_threshold: float,
) -> Tuple[Image.Image, List[Dict[str, Any]]]:
    """
    Run YOLO inference directly.

    This function makes app.py robust even if inference.py
    has a different function interface.

    Returns
    -------
    annotated_image
    detections
    """

    import numpy as np

    # --------------------------------------------------------
    # Convert image
    # --------------------------------------------------------

    image_rgb = image.convert(
        "RGB"
    )

    image_array = np.array(
        image_rgb
    )

    # --------------------------------------------------------
    # YOLO inference
    # --------------------------------------------------------

    results = model.predict(
        source=image_array,
        conf=confidence_threshold,
        verbose=False,
    )

    if not results:

        return (
            image_rgb,
            [],
        )

    result = results[0]

    # --------------------------------------------------------
    # Original dimensions
    # --------------------------------------------------------

    image_width, image_height = (
        image_rgb.size
    )

    # --------------------------------------------------------
    # Prepare annotated image
    # --------------------------------------------------------

    annotated = image_rgb.copy()

    draw = ImageDraw.Draw(
        annotated
    )

    detections = []

    # --------------------------------------------------------
    # No boxes
    # --------------------------------------------------------

    if result.boxes is None:

        return (
            annotated,
            detections,
        )

    # --------------------------------------------------------
    # Extract detections
    # --------------------------------------------------------

    boxes = result.boxes

    for index in range(
        len(boxes)
    ):

        # ----------------------------------------------------
        # Class
        # ----------------------------------------------------

        class_id = int(
            boxes.cls[index].item()
        )

        # ----------------------------------------------------
        # Confidence
        # ----------------------------------------------------

        confidence = float(
            boxes.conf[index].item()
        )

        # ----------------------------------------------------
        # Bounding box
        # ----------------------------------------------------

        xyxy = boxes.xyxy[
            index
        ].tolist()

        x1, y1, x2, y2 = [
            float(value)
            for value in xyxy
        ]

        # ----------------------------------------------------
        # Damage name
        # ----------------------------------------------------

        if hasattr(
            result,
            "names",
        ):

            damage_name = result.names.get(
                class_id,
                CLASS_NAMES.get(
                    class_id,
                    f"Class {class_id}",
                ),
            )

        else:

            damage_name = CLASS_NAMES.get(
                class_id,
                f"Class {class_id}",
            )

        # ----------------------------------------------------
        # Store detection
        # ----------------------------------------------------

        detections.append(
            {
                "damage_type":
                    damage_name,

                "class_id":
                    class_id,

                "confidence":
                    confidence,

                "bounding_box":
                    {
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                    },
            }
        )

        # ----------------------------------------------------
        # Draw bounding box
        # ----------------------------------------------------

        draw.rectangle(
            [
                x1,
                y1,
                x2,
                y2,
            ],
            outline="red",
            width=4,
        )

        # ----------------------------------------------------
        # Label
        # ----------------------------------------------------

        label = (
            f"{damage_name} "
            f"{confidence * 100:.1f}%"
        )

        # Approximate text size.
        text_box = draw.textbbox(
            (0, 0),
            label,
        )

        text_width = (
            text_box[2]
            - text_box[0]
        )

        text_height = (
            text_box[3]
            - text_box[1]
        )

        label_y = max(
            0,
            y1 - text_height - 8,
        )

        # Background
        draw.rectangle(
            [
                x1,
                label_y,
                x1 + text_width + 10,
                label_y + text_height + 8,
            ],
            fill="red",
        )

        # Text
        draw.text(
            (
                x1 + 5,
                label_y + 4,
            ),
            label,
            fill="white",
        )

    return (
        annotated,
        detections,
    )


# ============================================================
# SEVERITY ENGINE
# ============================================================

def calculate_fallback_severity(
    detections: List[Dict[str, Any]],
    image_width: int,
    image_height: int,
) -> Dict[str, Any]:
    """
    Fallback severity calculation.

    Normally severity.py is used.

    This fallback keeps the application functional if
    severity.py cannot be imported.
    """

    results = []

    damage_weights = {
        "Longitudinal Crack": 0.70,
        "Transverse Crack": 0.65,
        "Alligator Crack": 0.85,
        "Other Corruption": 0.50,
        "Pothole": 1.00,
    }

    total_image_area = max(
        1,
        image_width * image_height,
    )

    for index, detection in enumerate(
        detections,
        start=1,
    ):

        box = detection[
            "bounding_box"
        ]

        x1 = box["x1"]
        y1 = box["y1"]
        x2 = box["x2"]
        y2 = box["y2"]

        box_area = max(
            0,
            x2 - x1,
        ) * max(
            0,
            y2 - y1,
        )

        area_ratio = (
            box_area
            /
            total_image_area
        )

        area_score = clamp(
            area_ratio
            /
            0.10
            *
            100
        )

        confidence = clamp(
            detection[
                "confidence"
            ]
            *
            100
        )

        damage_name = detection[
            "damage_type"
        ]

        damage_score = (
            damage_weights.get(
                damage_name,
                0.50,
            )
            *
            100
        )

        severity_score = clamp(
            (
                0.70
                * area_score
            )
            +
            (
                0.30
                * confidence
            )
        )

        risk_score = clamp(
            (
                0.45
                * damage_score
            )
            +
            (
                0.30
                * confidence
            )
            +
            (
                0.25
                * area_score
            )
        )

        if severity_score >= 60:
            severity = "HIGH"

        elif severity_score >= 30:
            severity = "MEDIUM"

        else:
            severity = "LOW"

        if risk_score >= 70:
            priority = "HIGH"

        elif risk_score >= 40:
            priority = "MEDIUM"

        else:
            priority = "LOW"

        if priority == "HIGH":

            recommendation = (
                "Schedule prompt field inspection "
                "and assess the affected road section."
            )

        elif priority == "MEDIUM":

            recommendation = (
                "Schedule inspection and monitor "
                "the affected road section."
            )

        else:

            recommendation = (
                "Continue monitoring during future "
                "road inspections."
            )

        results.append(
            {
                "id": index,
                "damage_type": damage_name,
                "confidence": detection["confidence"],
                "confidence_percentage": confidence,
                "bounding_box": box,
                "area_ratio": area_ratio,
                "area_percentage":
                    area_ratio * 100,
                "area_score": area_score,
                "severity_score":
                    round(
                        severity_score,
                        2,
                    ),
                "severity": severity,
                "risk_score":
                    round(
                        risk_score,
                        2,
                    ),
                "priority": priority,
                "recommendation":
                    recommendation,
                "explanation":
                    (
                        f"{damage_name} detected with "
                        f"{confidence:.1f}% confidence."
                    ),
            }
        )

    if not results:

        return {
            "detections_count": 0,
            "damage_types_detected": [],
            "overall_risk_score": 0,
            "overall_severity_score": 0,
            "overall_severity": "LOW",
            "overall_priority": "LOW",
            "priority_counts": {
                "HIGH": 0,
                "MEDIUM": 0,
                "LOW": 0,
            },
            "highest_risk_detection": None,
            "highest_severity_detection": None,
            "recommendation":
                "No road damage was detected.",
            "detections": [],
        }

    highest_risk = max(
        results,
        key=lambda x: x[
            "risk_score"
        ],
    )

    highest_severity = max(
        results,
        key=lambda x: x[
            "severity_score"
        ],
    )

    overall_risk = highest_risk[
        "risk_score"
    ]

    overall_severity_score = (
        highest_severity[
            "severity_score"
        ]
    )

    if overall_risk >= 70:
        overall_priority = "HIGH"

    elif overall_risk >= 40:
        overall_priority = "MEDIUM"

    else:
        overall_priority = "LOW"

    if overall_severity_score >= 60:
        overall_severity = "HIGH"

    elif overall_severity_score >= 30:
        overall_severity = "MEDIUM"

    else:
        overall_severity = "LOW"

    return {
        "detections_count":
            len(results),

        "damage_types_detected":
            sorted(
                set(
                    x["damage_type"]
                    for x in results
                )
            ),

        "overall_risk_score":
            overall_risk,

        "overall_severity_score":
            overall_severity_score,

        "overall_severity":
            overall_severity,

        "overall_priority":
            overall_priority,

        "priority_counts": {
            "HIGH":
                sum(
                    x["priority"] == "HIGH"
                    for x in results
                ),
            "MEDIUM":
                sum(
                    x["priority"] == "MEDIUM"
                    for x in results
                ),
            "LOW":
                sum(
                    x["priority"] == "LOW"
                    for x in results
                ),
        },

        "highest_risk_detection":
            highest_risk,

        "highest_severity_detection":
            highest_severity,

        "recommendation":
            (
                "High-priority road damage detected. "
                "Consider prompt field inspection."
                if overall_priority == "HIGH"
                else
                "Moderate-priority damage detected. "
                "Schedule inspection and monitoring."
                if overall_priority == "MEDIUM"
                else
                "Low-priority visual damage detected. "
                "Continue monitoring."
            ),

        "detections":
            results,
    }


# ============================================================
# RUN SEVERITY.PY
# ============================================================

def run_severity_engine(
    detections: List[Dict[str, Any]],
    image_width: int,
    image_height: int,
) -> Dict[str, Any]:
    """
    Try to use the project's severity.py.

    If unavailable or incompatible, use the internal
    fallback implementation.
    """

    try:

        from severity import analyze_detections

        return analyze_detections(
            detections=detections,
            image_width=image_width,
            image_height=image_height,
        )

    except Exception:

        return calculate_fallback_severity(
            detections=detections,
            image_width=image_width,
            image_height=image_height,
        )


# ============================================================
# STATUS HTML
# ============================================================

def status_html(
    value: str,
) -> str:

    value = str(
        value
    ).upper()

    if value == "HIGH":

        return (
            '<div class="status-high">'
            '🔴 HIGH'
            '</div>'
        )

    if value == "MEDIUM":

        return (
            '<div class="status-medium">'
            '🟡 MEDIUM'
            '</div>'
        )

    return (
        '<div class="status-low">'
        '🟢 LOW'
        '</div>'
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "⚙️ RoadGuard Controls"
    )

    st.markdown(
        "### Model"
    )

    confidence_threshold = st.slider(
        "Detection confidence",
        min_value=0.10,
        max_value=0.95,
        value=0.35,
        step=0.05,
        help=(
            "Only detections above this confidence "
            "threshold will be displayed."
        ),
    )

    st.markdown(
        "---"
    )

    st.markdown(
        "### Project Information"
    )

    st.write(
        f"**Dataset:** {DATASET_NAME}"
    )

    st.write(
        f"**Model:** {MODEL_NAME}"
    )

    st.write(
        "**Classes:** 5"
    )

    st.write(
        "**Application:** RoadGuard"
    )

    st.markdown(
        "---"
    )

    st.markdown(
        "### Model Location"
    )

    st.code(
        str(
            MODEL_PATH
        ),
        language="text",
    )

    if MODEL_PATH.exists():

        st.success(
            "✅ Model found"
        )

    else:

        st.error(
            "❌ Model not found"
        )

    st.markdown(
        "---"
    )

    st.caption(
        "RoadGuard uses one dataset: RDD2022."
    )

    st.caption(
        "Risk and severity scores are engineered "
        "decision-support signals, not accident "
        "probabilities."
    )


# ============================================================
# MODEL STATUS
# ============================================================

if not MODEL_PATH.exists():

    st.error(
        """
        ### ❌ Trained model not found

        Put your trained model file:

        `roadguard_best.pt`

        in the same folder as `app.py`.

        Your project should look like:

        ```
        RoadGuard/
        ├── app.py
        ├── inference.py
        ├── severity.py
        ├── roadguard_best.pt
        ├── requirements.txt
        └── README.md
        ```
        """
    )

    st.stop()


# ============================================================
# LOAD MODEL
# ============================================================

try:

    with st.spinner(
        "Loading RoadGuard AI model..."
    ):

        model = load_model(
            str(
                MODEL_PATH
            )
        )

except Exception as error:

    st.error(
        "Unable to load the RoadGuard model."
    )

    st.code(
        str(error)
    )

    st.info(
        "Make sure ultralytics is installed and "
        "roadguard_best.pt is a valid YOLO model."
    )

    st.stop()


# ============================================================
# MODEL READY
# ============================================================

st.success(
    "🟢 RoadGuard AI model is ready."
)


# ============================================================
# UPLOAD SECTION
# ============================================================

st.markdown(
    "## 📷 Upload Road Image"
)

st.write(
    """
    Upload a road image and RoadGuard will detect visible
    road damage, estimate an engineered severity score,
    calculate a prioritization score, and provide an
    inspection recommendation.
    """
)


uploaded_file = st.file_uploader(
    "Choose a road image",
    type=SUPPORTED_IMAGE_TYPES,
    help=(
        "Supported formats: JPG, JPEG, PNG and WEBP."
    ),
)


# ============================================================
# MAIN APPLICATION
# ============================================================

if uploaded_file is None:

    st.info(
        "👆 Upload a road image to start the analysis."
    )

    # --------------------------------------------------------
    # INFORMATION CARDS
    # --------------------------------------------------------

    st.markdown(
        "## 🧠 How RoadGuard Works"
    )

    col1, col2, col3, col4 = st.columns(
        4
    )

    with col1:

        st.markdown(
            """
            ### 1️⃣ Upload

            Provide a road image.
            """
        )

    with col2:

        st.markdown(
            """
            ### 2️⃣ Detect

            YOLO identifies visible
            road damage.
            """
        )

    with col3:

        st.markdown(
            """
            ### 3️⃣ Assess

            RoadGuard calculates
            severity and risk.
            """
        )

    with col4:

        st.markdown(
            """
            ### 4️⃣ Prioritize

            The system produces
            an inspection priority.
            """
        )

    st.stop()


# ============================================================
# LOAD IMAGE
# ============================================================

try:

    image = Image.open(
        uploaded_file
    ).convert(
        "RGB"
    )

except Exception as error:

    st.error(
        "Unable to read the uploaded image."
    )

    st.code(
        str(error)
    )

    st.stop()


# ============================================================
# IMAGE INFORMATION
# ============================================================

image_width, image_height = (
    image.size
)


# ============================================================
# PREVIEW
# ============================================================

st.markdown(
    "## 🖼️ Uploaded Image"
)

image_col1, image_col2 = st.columns(
    [3, 1]
)

with image_col1:

    st.image(
        image,
        caption="Input road image",
        use_container_width=True,
    )

with image_col2:

    st.metric(
        "Width",
        f"{image_width}px",
    )

    st.metric(
        "Height",
        f"{image_height}px",
    )

    st.metric(
        "Format",
        uploaded_file.type
        or "Image",
    )


# ============================================================
# ANALYZE BUTTON
# ============================================================

st.markdown(
    "---"
)

analyze_button = st.button(
    "🔍 Analyze Road",
    type="primary",
    use_container_width=True,
)


# ============================================================
# RUN ANALYSIS
# ============================================================

if analyze_button:

    # --------------------------------------------------------
    # YOLO
    # --------------------------------------------------------

    with st.spinner(
        "Running RoadGuard computer vision..."
    ):

        try:

            annotated_image, detections = (
                run_yolo_inference(
                    model=model,
                    image=image,
                    confidence_threshold=
                        confidence_threshold,
                )
            )

        except Exception as error:

            st.error(
                "Inference failed."
            )

            st.code(
                traceback.format_exc()
            )

            st.stop()

    # --------------------------------------------------------
    # Severity
    # --------------------------------------------------------

    with st.spinner(
        "Calculating severity and priority..."
    ):

        try:

            analysis = run_severity_engine(
                detections=detections,
                image_width=image_width,
                image_height=image_height,
            )

        except Exception as error:

            st.error(
                "Severity analysis failed."
            )

            st.code(
                traceback.format_exc()
            )

            st.stop()

    # ========================================================
    # SAVE IN SESSION
    # ========================================================

    st.session_state[
        "annotated_image"
    ] = annotated_image

    st.session_state[
        "analysis"
    ] = analysis

    st.session_state[
        "detections"
    ] = detections

    st.session_state[
        "original_image"
    ] = image

    st.session_state[
        "analysis_complete"
    ] = True


# ============================================================
# DISPLAY RESULTS
# ============================================================

if st.session_state.get(
    "analysis_complete",
    False,
):

    annotated_image = st.session_state[
        "annotated_image"
    ]

    analysis = st.session_state[
        "analysis"
    ]

    detections = st.session_state[
        "detections"
    ]

    # ========================================================
    # RESULTS HEADER
    # ========================================================

    st.markdown(
        "---"
    )

    st.markdown(
        "## 📊 RoadGuard Analysis Results"
    )

    # ========================================================
    # TOP METRICS
    # ========================================================

    metric1, metric2, metric3, metric4 = (
        st.columns(4)
    )

    with metric1:

        st.metric(
            "Damage Detected",
            analysis.get(
                "detections_count",
                len(detections),
            ),
        )

    with metric2:

        st.metric(
            "Risk Score",
            (
                f"{analysis.get('overall_risk_score', 0):.1f}"
                "/100"
            ),
        )

    with metric3:

        st.metric(
            "Severity",
            analysis.get(
                "overall_severity",
                "LOW",
            ),
        )

    with metric4:

        st.metric(
            "Priority",
            analysis.get(
                "overall_priority",
                "LOW",
            ),
        )

    # ========================================================
    # STATUS
    # ========================================================

    st.markdown(
        "### 🚦 Overall Road Status"
    )

    status_col1, status_col2 = st.columns(
        2
    )

    with status_col1:

        st.markdown(
            status_html(
                analysis.get(
                    "overall_severity",
                    "LOW",
                )
            ),
            unsafe_allow_html=True,
        )

    with status_col2:

        st.markdown(
            status_html(
                analysis.get(
                    "overall_priority",
                    "LOW",
                )
            ),
            unsafe_allow_html=True,
        )

    # ========================================================
    # ANNOTATED IMAGE
    # ========================================================

    st.markdown(
        "## 🎯 Detected Road Damage"
    )

    st.image(
        annotated_image,
        caption=(
            "RoadGuard detections"
        ),
        use_container_width=True,
    )

    # ========================================================
    # RECOMMENDATION
    # ========================================================

    st.markdown(
        "## 🛠️ Maintenance Recommendation"
    )

    recommendation = analysis.get(
        "recommendation",
        "No recommendation available.",
    )

    if analysis.get(
        "overall_priority"
    ) == "HIGH":

        st.error(
            f"🚨 {recommendation}"
        )

    elif analysis.get(
        "overall_priority"
    ) == "MEDIUM":

        st.warning(
            f"⚠️ {recommendation}"
        )

    else:

        st.success(
            f"✅ {recommendation}"
        )

    # ========================================================
    # DAMAGE TYPES
    # ========================================================

    damage_types = analysis.get(
        "damage_types_detected",
        [],
    )

    st.markdown(
        "### 🔎 Damage Types Detected"
    )

    if damage_types:

        type_columns = st.columns(
            min(
                len(damage_types),
                4,
            )
        )

        for column, damage in zip(
            type_columns,
            damage_types,
        ):

            with column:

                st.info(
                    f"🔧 {damage}"
                )

    else:

        st.success(
            "No visible road damage detected."
        )

    # ========================================================
    # PRIORITY DISTRIBUTION
    # ========================================================

    st.markdown(
        "## 📈 Priority Distribution"
    )

    priority_counts = analysis.get(
        "priority_counts",
        {
            "HIGH": 0,
            "MEDIUM": 0,
            "LOW": 0,
        },
    )

    chart_data = pd.DataFrame(
        {
            "Priority": [
                "HIGH",
                "MEDIUM",
                "LOW",
            ],
            "Detections": [
                priority_counts.get(
                    "HIGH",
                    0,
                ),
                priority_counts.get(
                    "MEDIUM",
                    0,
                ),
                priority_counts.get(
                    "LOW",
                    0,
                ),
            ],
        }
    )

    st.bar_chart(
        chart_data.set_index(
            "Priority"
        )
    )

    # ========================================================
    # DETECTION TABLE
    # ========================================================

    st.markdown(
        "## 📋 Detection Details"
    )

    detection_rows = []

    for detection in analysis.get(
        "detections",
        [],
    ):

        detection_rows.append(
            {
                "ID":
                    detection.get(
                        "id",
                        "",
                    ),

                "Damage Type":
                    detection.get(
                        "damage_type",
                        "",
                    ),

                "Confidence":
                    f"{detection.get('confidence_percentage', 0):.1f}%",

                "Area":
                    f"{detection.get('area_percentage', 0):.2f}%",

                "Severity":
                    detection.get(
                        "severity",
                        "LOW",
                    ),

                "Severity Score":
                    round(
                        safe_float(
                            detection.get(
                                "severity_score",
                                0,
                            )
                        ),
                        1,
                    ),

                "Risk":
                    round(
                        safe_float(
                            detection.get(
                                "risk_score",
                                0,
                            )
                        ),
                        1,
                    ),

                "Priority":
                    detection.get(
                        "priority",
                        "LOW",
                    ),
            }
        )

    if detection_rows:

        dataframe = pd.DataFrame(
            detection_rows
        )

        st.dataframe(
            dataframe,
            use_container_width=True,
            hide_index=True,
        )

    else:

        st.info(
            "No detections passed the confidence threshold."
        )

    # ========================================================
    # DETAILED DETECTION CARDS
    # ========================================================

    st.markdown(
        "## 🔬 Detailed Detection Analysis"
    )

    detailed_detections = analysis.get(
        "detections",
        [],
    )

    if detailed_detections:

        for detection in detailed_detections:

            damage = detection.get(
                "damage_type",
                "Unknown",
            )

            priority = detection.get(
                "priority",
                "LOW",
            )

            with st.expander(
                (
                    f"Detection #{detection.get('id', '?')} — "
                    f"{damage} — {priority}"
                )
            ):

                col_a, col_b, col_c = (
                    st.columns(3)
                )

                with col_a:

                    st.metric(
                        "Confidence",
                        (
                            f"{detection.get('confidence_percentage', 0):.1f}%"
                        ),
                    )

                with col_b:

                    st.metric(
                        "Severity Score",
                        (
                            f"{detection.get('severity_score', 0):.1f}/100"
                        ),
                    )

                with col_c:

                    st.metric(
                        "Risk Score",
                        (
                            f"{detection.get('risk_score', 0):.1f}/100"
                        ),
                    )

                st.markdown(
                    "### Severity"
                )

                st.write(
                    detection.get(
                        "severity",
                        "LOW",
                    )
                )

                st.markdown(
                    "### Priority"
                )

                st.write(
                    detection.get(
                        "priority",
                        "LOW",
                    )
                )

                st.markdown(
                    "### Recommendation"
                )

                st.write(
                    detection.get(
                        "recommendation",
                        "No recommendation.",
                    )
                )

                st.markdown(
                    "### Explanation"
                )

                st.write(
                    detection.get(
                        "explanation",
                        "",
                    )
                )

                st.markdown(
                    "### Bounding Box"
                )

                st.json(
                    detection.get(
                        "bounding_box",
                        {},
                    )
                )

    # ========================================================
    # HIGHEST RISK
    # ========================================================

    highest_risk = analysis.get(
        "highest_risk_detection"
    )

    if highest_risk:

        st.markdown(
            "## 🚨 Highest Priority Detection"
        )

        hr_col1, hr_col2 = st.columns(
            2
        )

        with hr_col1:

            st.metric(
                "Damage",
                highest_risk.get(
                    "damage_type",
                    "Unknown",
                ),
            )

            st.metric(
                "Risk Score",
                (
                    f"{highest_risk.get('risk_score', 0):.1f}/100"
                ),
            )

        with hr_col2:

            st.metric(
                "Severity",
                highest_risk.get(
                    "severity",
                    "LOW",
                ),
            )

            st.metric(
                "Priority",
                highest_risk.get(
                    "priority",
                    "LOW",
                ),
            )

    # ========================================================
    # JSON REPORT
    # ========================================================

    st.markdown(
        "## 📄 Export Analysis"
    )

    report = {
        "application": APP_NAME,
        "version": APP_VERSION,
        "dataset": DATASET_NAME,
        "model": MODEL_NAME,
        "model_file": MODEL_PATH.name,
        "image": {
            "filename":
                uploaded_file.name,
            "width":
                image_width,
            "height":
                image_height,
        },
        "confidence_threshold":
            confidence_threshold,
        "analysis":
            analysis,
    }

    json_data = json.dumps(
        report,
        indent=4,
        ensure_ascii=False,
    )

    download_col1, download_col2 = (
        st.columns(2)
    )

    with download_col1:

        st.download_button(
            label="⬇️ Download JSON Report",
            data=json_data,
            file_name=(
                "roadguard_report.json"
            ),
            mime="application/json",
            use_container_width=True,
        )

    # ========================================================
    # ANNOTATED IMAGE DOWNLOAD
    # ========================================================

    with download_col2:

        image_bytes = io.BytesIO()

        annotated_image.save(
            image_bytes,
            format="PNG",
        )

        image_bytes.seek(0)

        st.download_button(
            label="⬇️ Download Annotated Image",
            data=image_bytes,
            file_name=(
                "roadguard_annotated.png"
            ),
            mime="image/png",
            use_container_width=True,
        )

    # ========================================================
    # RAW JSON
    # ========================================================

    with st.expander(
        "🔧 View complete JSON result"
    ):

        st.json(
            report
        )


# ============================================================
# ABOUT SECTION
# ============================================================

st.markdown(
    "---"
)

with st.expander(
    "ℹ️ About RoadGuard"
):

    st.markdown(
        f"""
        ### 🛣️ RoadGuard Infrastructure Intelligence

        **RoadGuard** is a computer-vision based road
        infrastructure analysis system.

        **Pipeline**

        ```
        Road Image
             ↓
        YOLOv8n
             ↓
        Road Damage Detection
             ↓
        Severity Engine
             ↓
        Risk Score
             ↓
        Maintenance Priority
             ↓
        Recommendation
        ```

        **Single Dataset**

        `{DATASET_NAME}`

        **Detection Model**

        `{MODEL_NAME}`

        **Detected Classes**

        - Longitudinal Crack
        - Transverse Crack
        - Alligator Crack
        - Other Corruption
        - Pothole

        **Important**

        The severity and risk values shown by RoadGuard are
        engineered decision-support scores based on the
        detected damage type, model confidence and visible
        bounding-box area.

        They should not be interpreted as calibrated
        probabilities of accidents or guaranteed engineering
        maintenance requirements.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    f"""
    <div class="roadguard-footer">

        <strong>{APP_NAME}</strong><br>

        Computer Vision • YOLOv8n • RDD2022<br>

        Version {APP_VERSION}

    </div>
    """,
    unsafe_allow_html=True,
)
