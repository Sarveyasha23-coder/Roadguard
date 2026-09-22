"""
============================================================
ROADGUARD INFRASTRUCTURE INTELLIGENCE
============================================================

AI-Powered Road Damage Detection, Severity Assessment
and Infrastructure Maintenance Prioritization

Dataset:
    RDD2022

Model:
    YOLOv8n

Detection Classes:
    1. Longitudinal Crack
    2. Transverse Crack
    3. Alligator Crack
    4. Other Corruption
    5. Pothole

Main Features:
    - Road damage image upload
    - YOLOv8n object detection
    - Confidence threshold control
    - Bounding-box visualization
    - Damage severity estimation
    - Infrastructure risk assessment
    - Maintenance priority
    - Detection statistics
    - Detection table
    - JSON report download
    - Annotated image download
    - Streamlit dashboard

Project Structure:

    Roadguard/
    ├── app.py
    ├── inference.py
    ├── severity.py
    ├── requirements.txt
    ├── roadguard_best.pt
    └── README.md

============================================================
"""

from __future__ import annotations

import io
import json
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="RoadGuard Infrastructure Intelligence",
    page_icon="🚧",
    layout="wide",
    initial_sidebar_state="expanded",
)


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
# PAGE STYLE
# ============================================================

st.markdown(
    """
    <style>

    /* -----------------------------------------------------
       Main application
    ----------------------------------------------------- */

    .main {
        background-color: #f7f8fa;
    }

    /* -----------------------------------------------------
       Header
    ----------------------------------------------------- */

    .roadguard-header {
        background: linear-gradient(
            135deg,
            #111827 0%,
            #1f2937 55%,
            #374151 100%
        );

        padding: 30px;
        border-radius: 18px;
        margin-bottom: 25px;
        color: white;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.12);
    }

    .roadguard-title {
        font-size: 38px;
        font-weight: 800;
        margin-bottom: 8px;
    }

    .roadguard-subtitle {
        font-size: 17px;
        line-height: 1.6;
        opacity: 0.9;
    }

    /* -----------------------------------------------------
       Cards
    ----------------------------------------------------- */

    .metric-card {
        background: white;
        border-radius: 15px;
        padding: 20px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.06);
        min-height: 120px;
    }

    .metric-title {
        font-size: 14px;
        color: #6b7280;
        font-weight: 600;
        margin-bottom: 8px;
    }

    .metric-value {
        font-size: 30px;
        font-weight: 800;
        color: #111827;
    }

    .metric-description {
        font-size: 12px;
        color: #6b7280;
        margin-top: 5px;
    }

    /* -----------------------------------------------------
       Severity cards
    ----------------------------------------------------- */

    .severity-card {
        border-radius: 15px;
        padding: 22px;
        color: white;
        margin-top: 10px;
        margin-bottom: 10px;
    }

    .severity-critical {
        background: linear-gradient(
            135deg,
            #7f1d1d,
            #dc2626
        );
    }

    .severity-high {
        background: linear-gradient(
            135deg,
            #9a3412,
            #ea580c
        );
    }

    .severity-medium {
        background: linear-gradient(
            135deg,
            #854d0e,
            #ca8a04
        );
    }

    .severity-low {
        background: linear-gradient(
            135deg,
            #166534,
            #16a34a
        );
    }

    .severity-title {
        font-size: 24px;
        font-weight: 800;
    }

    .severity-text {
        font-size: 14px;
        margin-top: 8px;
        line-height: 1.5;
    }

    /* -----------------------------------------------------
       Information boxes
    ----------------------------------------------------- */

    .info-box {
        background: #eff6ff;
        border-left: 5px solid #2563eb;
        padding: 15px;
        border-radius: 10px;
        margin-top: 10px;
        margin-bottom: 15px;
    }

    .success-box {
        background: #f0fdf4;
        border-left: 5px solid #16a34a;
        padding: 15px;
        border-radius: 10px;
        margin-top: 10px;
        margin-bottom: 15px;
    }

    .warning-box {
        background: #fffbeb;
        border-left: 5px solid #d97706;
        padding: 15px;
        border-radius: 10px;
        margin-top: 10px;
        margin-bottom: 15px;
    }

    /* -----------------------------------------------------
       Footer
    ----------------------------------------------------- */

    .footer {
        margin-top: 50px;
        padding: 20px;
        text-align: center;
        color: #6b7280;
        font-size: 13px;
        border-top: 1px solid #e5e7eb;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# IMPORT INFERENCE ENGINE
# ============================================================

try:

    from inference import (
        get_model,
        predict_image,
        summarize_detections,
        annotate_image,
        get_model_info,
    )

    INFERENCE_IMPORT_ERROR = None

except Exception as exc:

    get_model = None
    predict_image = None
    summarize_detections = None
    annotate_image = None
    get_model_info = None

    INFERENCE_IMPORT_ERROR = exc


# ============================================================
# OPTIONAL SEVERITY ENGINE
# ============================================================

SEVERITY_ENGINE_AVAILABLE = False
severity_engine = None

try:

    import severity as severity_module

    severity_engine = severity_module

    SEVERITY_ENGINE_AVAILABLE = True

except Exception:

    severity_engine = None
    SEVERITY_ENGINE_AVAILABLE = False


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="roadguard-header">

        <div class="roadguard-title">
            🚧 RoadGuard Infrastructure Intelligence
        </div>

        <div class="roadguard-subtitle">
            AI-powered road damage detection, severity assessment,
            infrastructure risk analysis and maintenance prioritization.
        </div>

    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "## ⚙️ RoadGuard Controls"
    )

    st.markdown(
        "---"
    )

    st.markdown(
        "### 🤖 Model"
    )

    confidence_threshold = st.slider(
        "Detection confidence",
        min_value=0.05,
        max_value=0.95,
        value=0.35,
        step=0.05,
        help=(
            "Minimum confidence required for a detection "
            "to be displayed."
        ),
    )

    iou_threshold = st.slider(
        "IoU threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.45,
        step=0.05,
        help=(
            "Intersection-over-Union threshold used by YOLO "
            "for non-maximum suppression."
        ),
    )

    image_size = st.selectbox(
        "Inference image size",
        options=[
            320,
            416,
            512,
            640,
            768,
        ],
        index=3,
        help="Input image size used by the YOLO model.",
    )

    st.markdown(
        "---"
    )

    st.markdown(
        "### 📊 Project Information"
    )

    st.write(
        "**Dataset:** RDD2022"
    )

    st.write(
        "**Model:** YOLOv8n"
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
        "### 📁 Model Location"
    )

    st.code(
        str(MODEL_PATH),
        language="text",
    )


# ============================================================
# MODEL STATUS
# ============================================================

if not MODEL_PATH.exists():

    st.error(
        "🚨 RoadGuard model file was not found."
    )

    st.warning(
        f"""
Expected model location:

{MODEL_PATH}

Your GitHub repository should contain:

Roadguard/
├── app.py
├── inference.py
├── severity.py
├── requirements.txt
├── roadguard_best.pt
└── README.md
"""
    )

    st.stop()


# ============================================================
# INFERENCE ENGINE STATUS
# ============================================================

if INFERENCE_IMPORT_ERROR is not None:

    st.error(
        "Unable to initialize the RoadGuard inference engine."
    )

    st.code(
        (
            f"{type(INFERENCE_IMPORT_ERROR).__name__}: "
            f"{INFERENCE_IMPORT_ERROR}"
        ),
        language="text",
    )

    st.warning(
        """
Please check the Streamlit deployment logs.

The detailed error above is intentionally shown so that
dependency or Ultralytics problems are not hidden.
"""
    )

    st.stop()


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_roadguard_model():

    return get_model(
        model_path=MODEL_PATH
    )


try:

    model = load_roadguard_model()

except Exception as exc:

    st.error(
        "❌ Unable to load the RoadGuard model."
    )

    st.error(
        f"{type(exc).__name__}: {exc}"
    )

    with st.expander(
        "🔍 Technical error details"
    ):

        st.code(
            traceback.format_exc(),
            language="text",
        )

    st.warning(
        """
Make sure:

1. roadguard_best.pt is present in the repository.
2. The file is a valid YOLO model.
3. Ultralytics is installed.
4. PyTorch is installed correctly.
5. The Streamlit deployment finished installing
   all requirements.
"""
    )

    st.stop()


# ============================================================
# MODEL READY
# ============================================================

st.success(
    "✅ RoadGuard YOLOv8n model loaded successfully."
)


# ============================================================
# UPLOAD SECTION
# ============================================================

st.markdown(
    "## 📷 Road Image Analysis"
)

st.markdown(
    """
Upload a road image and RoadGuard will detect visible road
damage, estimate severity and generate a maintenance priority
assessment.
"""
)

uploaded_file = st.file_uploader(
    "Upload a road image",
    type=[
        "jpg",
        "jpeg",
        "png",
        "webp",
        "bmp",
    ],
    help=(
        "Supported formats: JPG, JPEG, PNG, WEBP and BMP."
    ),
)


# ============================================================
# NO IMAGE
# ============================================================

if uploaded_file is None:

    st.markdown(
        """
        <div class="info-box">

        <strong>👆 Upload an image to begin.</strong>

        <br><br>

        RoadGuard accepts road images and uses a lightweight
        YOLOv8n model trained on the RDD2022 road-damage dataset.

        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        "## 🧠 Detection Classes"
    )

    class_columns = st.columns(5)

    for index, (
        class_id,
        class_name,
    ) in enumerate(CLASS_NAMES.items()):

        with class_columns[index]:

            st.markdown(
                f"""
                <div class="metric-card">

                <div class="metric-title">
                    Class {class_id}
                </div>

                <div style="
                    font-size:16px;
                    font-weight:700;
                    color:#111827;
                ">
                    {class_name}
                </div>

                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(
        """
        <div class="footer">

        RoadGuard Infrastructure Intelligence<br>
        RDD2022 • YOLOv8n • Computer Vision

        </div>
        """,
        unsafe_allow_html=True,
    )

    st.stop()


# ============================================================
# LOAD IMAGE
# ============================================================

try:

    original_image = Image.open(
        uploaded_file
    ).convert("RGB")

except Exception as exc:

    st.error(
        "Unable to read the uploaded image."
    )

    st.code(
        f"{type(exc).__name__}: {exc}",
        language="text",
    )

    st.stop()


# ============================================================
# IMAGE INFORMATION
# ============================================================

image_width, image_height = (
    original_image.size
)

image_array = np.asarray(
    original_image
)


# ============================================================
# RUN INFERENCE BUTTON
# ============================================================

st.markdown(
    "### 🔍 Detection"
)

analyze_button = st.button(
    "🚀 Analyze Road Damage",
    type="primary",
    use_container_width=True,
)


# ============================================================
# AUTOMATIC ANALYSIS WHEN IMAGE IS UPLOADED
# ============================================================

if analyze_button:

    with st.spinner(
        "Running RoadGuard AI analysis..."
    ):

        try:

            detections = predict_image(
                image=image_array,
                confidence=confidence_threshold,
                iou=iou_threshold,
                image_size=image_size,
                model_path=MODEL_PATH,
            )

        except Exception as exc:

            st.error(
                "❌ RoadGuard inference failed."
            )

            st.error(
                f"{type(exc).__name__}: {exc}"
            )

            with st.expander(
                "🔍 Full technical error"
            ):

                st.code(
                    traceback.format_exc(),
                    language="text",
                )

            st.stop()

    # ========================================================
    # SAVE RESULTS IN SESSION STATE
    # ========================================================

    st.session_state["roadguard_detections"] = (
        detections
    )

    st.session_state["roadguard_image"] = (
        original_image
    )

    st.session_state["roadguard_filename"] = (
        uploaded_file.name
    )

    st.session_state["roadguard_analysis_time"] = (
        datetime.now().isoformat()
    )

    st.success(
        "✅ Analysis completed successfully."
    )


# ============================================================
# CHECK IF RESULTS EXIST
# ============================================================

if (
    "roadguard_detections"
    not in st.session_state
):

    st.info(
        "Click **Analyze Road Damage** to start the AI analysis."
    )

    st.stop()


# ============================================================
# RETRIEVE RESULTS
# ============================================================

detections = st.session_state[
    "roadguard_detections"
]

analysis_image = st.session_state[
    "roadguard_image"
]

filename = st.session_state.get(
    "roadguard_filename",
    "road_image.jpg",
)


# ============================================================
# SUMMARY
# ============================================================

if summarize_detections is not None:

    summary = summarize_detections(
        detections
    )

else:

    confidence_values = [
        float(
            item.get(
                "confidence",
                0,
            )
        )
        for item in detections
    ]

    class_counts: Dict[str, int] = {}

    for item in detections:

        class_name = item.get(
            "class_name",
            "Unknown",
        )

        class_counts[class_name] = (
            class_counts.get(
                class_name,
                0,
            )
            + 1
        )

    summary = {
        "total_detections": len(detections),
        "average_confidence": (
            sum(confidence_values)
            / len(confidence_values)
            if confidence_values
            else 0
        ),
        "highest_confidence": (
            max(confidence_values)
            if confidence_values
            else 0
        ),
        "classes": class_counts,
    }


# ============================================================
# TOP METRICS
# ============================================================

st.markdown(
    "## 📊 Detection Overview"
)

total_detections = int(
    summary.get(
        "total_detections",
        len(detections),
    )
)

average_confidence = float(
    summary.get(
        "average_confidence",
        0,
    )
)

highest_confidence = float(
    summary.get(
        "highest_confidence",
        0,
    )
)


metric_columns = st.columns(4)


with metric_columns[0]:

    st.markdown(
        f"""
        <div class="metric-card">

        <div class="metric-title">
            Total Damage Detections
        </div>

        <div class="metric-value">
            {total_detections}
        </div>

        <div class="metric-description">
            Objects detected in the image
        </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


with metric_columns[1]:

    st.markdown(
        f"""
        <div class="metric-card">

        <div class="metric-title">
            Average Confidence
        </div>

        <div class="metric-value">
            {average_confidence:.1%}
        </div>

        <div class="metric-description">
            Mean detection confidence
        </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


with metric_columns[2]:

    st.markdown(
        f"""
        <div class="metric-card">

        <div class="metric-title">
            Highest Confidence
        </div>

        <div class="metric-value">
            {highest_confidence:.1%}
        </div>

        <div class="metric-description">
            Strongest detection
        </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


with metric_columns[3]:

    st.markdown(
        f"""
        <div class="metric-card">

        <div class="metric-title">
            Image Resolution
        </div>

        <div class="metric-value">
            {image_width}×{image_height}
        </div>

        <div class="metric-description">
            Uploaded image dimensions
        </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# CREATE ANNOTATED IMAGE
# ============================================================

try:

    annotated_array = annotate_image(
        image=np.asarray(
            analysis_image
        ),
        detections=detections,
    )

    annotated_image = Image.fromarray(
        annotated_array
    )

except Exception:

    # --------------------------------------------------------
    # PIL fallback annotation
    # --------------------------------------------------------

    annotated_image = analysis_image.copy()

    draw = ImageDraw.Draw(
        annotated_image
    )

    try:

        font = ImageFont.load_default()

    except Exception:

        font = None

    for detection in detections:

        bbox = detection.get(
            "bbox",
            {},
        )

        x1 = int(
            bbox.get("x1", 0)
        )

        y1 = int(
            bbox.get("y1", 0)
        )

        x2 = int(
            bbox.get("x2", 0)
        )

        y2 = int(
            bbox.get("y2", 0)
        )

        label = (
            f"{detection.get('class_name', 'Unknown')} "
            f"{float(detection.get('confidence', 0)):.2f}"
        )

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

        draw.text(
            [
                x1,
                max(0, y1 - 18),
            ],
            label,
            fill="red",
            font=font,
        )


# ============================================================
# IMAGE DISPLAY
# ============================================================

st.markdown(
    "## 🖼️ Detection Result"
)

image_col1, image_col2 = st.columns(2)


with image_col1:

    st.markdown(
        "### Original Image"
    )

    st.image(
        analysis_image,
        use_container_width=True,
    )


with image_col2:

    st.markdown(
        "### AI Detection"
    )

    st.image(
        annotated_image,
        use_container_width=True,
    )


# ============================================================
# SEVERITY ENGINE
# ============================================================

def calculate_severity(
    detections: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Calculate an interpretable RoadGuard severity assessment.

    This is an application-level assessment intended for
    prioritization and demonstration. It is not a structural
    engineering diagnosis.
    """

    if not detections:

        return {
            "severity": "Low",
            "risk_score": 0,
            "priority": "Routine Inspection",
            "recommendation": (
                "No road damage was detected above the "
                "selected confidence threshold."
            ),
        }

    # --------------------------------------------------------
    # Base severity weights
    # --------------------------------------------------------

    class_weights = {
        "Longitudinal Crack": 25,
        "Transverse Crack": 25,
        "Alligator Crack": 35,
        "Other Corruption": 20,
        "Pothole": 40,
    }

    weighted_scores = []

    for detection in detections:

        class_name = detection.get(
            "class_name",
            "Unknown",
        )

        confidence = float(
            detection.get(
                "confidence",
                0,
            )
        )

        area = float(
            detection.get(
                "area",
                0,
            )
        )

        # ----------------------------------------------------
        # Confidence contribution
        # ----------------------------------------------------

        base_weight = class_weights.get(
            class_name,
            20,
        )

        confidence_score = (
            base_weight
            * confidence
        )

        # ----------------------------------------------------
        # Bounding box area contribution
        # ----------------------------------------------------

        image_area = (
            image_width
            * image_height
        )

        if image_area > 0:

            area_ratio = (
                area / image_area
            )

        else:

            area_ratio = 0.0

        # Cap area contribution to avoid
        # extremely large scores.
        area_bonus = min(
            area_ratio * 100,
            20,
        )

        detection_score = (
            confidence_score
            + area_bonus
        )

        weighted_scores.append(
            detection_score
        )

    # --------------------------------------------------------
    # Aggregate risk
    # --------------------------------------------------------

    raw_score = sum(
        weighted_scores
    )

    # Multiple detections increase urgency.
    count_bonus = min(
        len(detections) * 5,
        25,
    )

    risk_score = min(
        100,
        int(
            raw_score
            + count_bonus
        ),
    )

    # --------------------------------------------------------
    # Severity level
    # --------------------------------------------------------

    if risk_score >= 75:

        severity = "Critical"
        priority = "Immediate Intervention"

        recommendation = (
            "Multiple or high-impact road defects were detected. "
            "The affected area should receive prompt field "
            "inspection and maintenance planning."
        )

    elif risk_score >= 50:

        severity = "High"
        priority = "High Priority"

        recommendation = (
            "Significant road damage was detected. "
            "A field inspection and maintenance assessment "
            "should be scheduled soon."
        )

    elif risk_score >= 25:

        severity = "Medium"
        priority = "Planned Maintenance"

        recommendation = (
            "Moderate road damage was detected. "
            "The location should be monitored and considered "
            "for planned maintenance."
        )

    else:

        severity = "Low"
        priority = "Routine Inspection"

        recommendation = (
            "Detected damage appears limited according to "
            "the current image and confidence threshold. "
            "Routine monitoring is recommended."
        )

    return {
        "severity": severity,
        "risk_score": risk_score,
        "priority": priority,
        "recommendation": recommendation,
    }


# ============================================================
# RUN SEVERITY ANALYSIS
# ============================================================

severity_result = calculate_severity(
    detections
)


severity_level = severity_result[
    "severity"
]

risk_score = severity_result[
    "risk_score"
]

priority = severity_result[
    "priority"
]

recommendation = severity_result[
    "recommendation"
]


# ============================================================
# SEVERITY DISPLAY
# ============================================================

st.markdown(
    "## 🚦 Infrastructure Risk Assessment"
)


severity_css_class = {
    "Critical": "severity-critical",
    "High": "severity-high",
    "Medium": "severity-medium",
    "Low": "severity-low",
}.get(
    severity_level,
    "severity-low",
)


severity_description = {
    "Critical": (
        "Very high-priority road damage indicators "
        "were detected."
    ),
    "High": (
        "Substantial road damage indicators "
        "were detected."
    ),
    "Medium": (
        "Moderate road damage indicators "
        "were detected."
    ),
    "Low": (
        "Limited road damage indicators "
        "were detected."
    ),
}.get(
    severity_level,
    "Road condition assessment completed.",
)


st.markdown(
    f"""
    <div class="severity-card {severity_css_class}">

        <div class="severity-title">
            {severity_level} Severity
        </div>

        <div style="
            font-size:32px;
            font-weight:800;
            margin-top:8px;
        ">
            Risk Score: {risk_score}/100
        </div>

        <div class="severity-text">
            {severity_description}
        </div>

    </div>
    """,
    unsafe_allow_html=True,
)


risk_col1, risk_col2, risk_col3 = st.columns(3)


with risk_col1:

    st.metric(
        "Severity",
        severity_level,
    )


with risk_col2:

    st.metric(
        "Risk Score",
        f"{risk_score}/100",
    )


with risk_col3:

    st.metric(
        "Maintenance Priority",
        priority,
    )


st.markdown(
    f"""
    <div class="info-box">

    <strong>Recommended Action</strong>

    <br><br>

    {recommendation}

    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DETECTION TABLE
# ============================================================

st.markdown(
    "## 🔎 Detailed Detections"
)


if detections:

    table_rows = []

    for index, detection in enumerate(
        detections,
        start=1,
    ):

        bbox = detection.get(
            "bbox",
            {},
        )

        table_rows.append(
            {
                "Detection": index,

                "Damage Type": detection.get(
                    "class_name",
                    "Unknown",
                ),

                "Confidence": (
                    f"{float(detection.get('confidence', 0)):.2%}"
                ),

                "X1": round(
                    float(
                        bbox.get(
                            "x1",
                            0,
                        )
                    ),
                    1,
                ),

                "Y1": round(
                    float(
                        bbox.get(
                            "y1",
                            0,
                        )
                    ),
                    1,
                ),

                "X2": round(
                    float(
                        bbox.get(
                            "x2",
                            0,
                        )
                    ),
                    1,
                ),

                "Y2": round(
                    float(
                        bbox.get(
                            "y2",
                            0,
                        )
                    ),
                    1,
                ),

                "Area": round(
                    float(
                        detection.get(
                            "area",
                            0,
                        )
                    ),
                    1,
                ),
            }
        )

    detection_dataframe = pd.DataFrame(
        table_rows
    )

    st.dataframe(
        detection_dataframe,
        use_container_width=True,
        hide_index=True,
    )

else:

    st.info(
        """
        No road damage was detected above the selected
        confidence threshold.
        """
    )


# ============================================================
# DAMAGE DISTRIBUTION
# ============================================================

st.markdown(
    "## 📈 Damage Distribution"
)


if detections:

    class_counts = {}

    for detection in detections:

        class_name = detection.get(
            "class_name",
            "Unknown",
        )

        class_counts[class_name] = (
            class_counts.get(
                class_name,
                0,
            )
            + 1
        )

    chart_dataframe = pd.DataFrame(
        {
            "Damage Type": list(
                class_counts.keys()
            ),
            "Detections": list(
                class_counts.values()
            ),
        }
    )

    chart_dataframe = (
        chart_dataframe
        .set_index(
            "Damage Type"
        )
    )

    st.bar_chart(
        chart_dataframe
    )

else:

    st.info(
        "No detection distribution is available."
    )


# ============================================================
# CLASS-BY-CLASS SUMMARY
# ============================================================

st.markdown(
    "## 🧩 Class Analysis"
)


class_columns = st.columns(
    len(CLASS_NAMES)
)


for index, (
    class_id,
    class_name,
) in enumerate(
    CLASS_NAMES.items()
):

    count = sum(
        1
        for detection in detections
        if int(
            detection.get(
                "class_id",
                -1,
            )
        ) == class_id
    )

    with class_columns[index]:

        st.markdown(
            f"""
            <div class="metric-card">

            <div class="metric-title">
                Class {class_id}
            </div>

            <div style="
                font-size:15px;
                font-weight:700;
                margin-bottom:10px;
            ">
                {class_name}
            </div>

            <div class="metric-value">
                {count}
            </div>

            <div class="metric-description">
                detections
            </div>

            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# REPORT GENERATION
# ============================================================

st.markdown(
    "## 📄 Analysis Report"
)


report = {
    "project": "RoadGuard Infrastructure Intelligence",

    "dataset": "RDD2022",

    "model": "YOLOv8n",

    "analysis_time": st.session_state.get(
        "roadguard_analysis_time",
        datetime.now().isoformat(),
    ),

    "image": {
        "filename": filename,
        "width": image_width,
        "height": image_height,
    },

    "settings": {
        "confidence_threshold": confidence_threshold,
        "iou_threshold": iou_threshold,
        "image_size": image_size,
    },

    "summary": {
        "total_detections": total_detections,
        "average_confidence": average_confidence,
        "highest_confidence": highest_confidence,
    },

    "severity": {
        "level": severity_level,
        "risk_score": risk_score,
        "priority": priority,
        "recommendation": recommendation,
    },

    "detections": detections,
}


# ============================================================
# JSON DOWNLOAD
# ============================================================

json_data = json.dumps(
    report,
    indent=4,
    default=str,
)


st.download_button(
    label="📥 Download JSON Report",
    data=json_data,
    file_name="roadguard_analysis_report.json",
    mime="application/json",
    use_container_width=True,
)


# ============================================================
# ANNOTATED IMAGE DOWNLOAD
# ============================================================

image_buffer = io.BytesIO()

annotated_image.save(
    image_buffer,
    format="PNG",
)

image_buffer.seek(0)


st.download_button(
    label="🖼️ Download Annotated Image",
    data=image_buffer,
    file_name="roadguard_annotated.png",
    mime="image/png",
    use_container_width=True,
)


# ============================================================
# MODEL INFORMATION
# ============================================================

with st.expander(
    "🤖 Model Information"
):

    if get_model_info is not None:

        try:

            model_info = get_model_info(
                MODEL_PATH
            )

            st.json(
                model_info
            )

        except Exception:

            st.json(
                {
                    "model": "YOLOv8n",
                    "dataset": "RDD2022",
                    "model_path": str(
                        MODEL_PATH
                    ),
                    "classes": CLASS_NAMES,
                }
            )

    else:

        st.json(
            {
                "model": "YOLOv8n",
                "dataset": "RDD2022",
                "model_path": str(
                    MODEL_PATH
                ),
                "classes": CLASS_NAMES,
            }
        )


# ============================================================
# SEVERITY DISCLAIMER
# ============================================================

st.markdown(
    """
    <div class="warning-box">

    <strong>⚠️ Important Note</strong>

    <br><br>

    RoadGuard provides computer-vision-based visual
    assessment and maintenance prioritization.

    The severity and risk score are intended as decision-support
    indicators and should not be treated as a replacement for
    professional civil-engineering inspection or structural
    assessment.

    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">

        <strong>🚧 RoadGuard Infrastructure Intelligence</strong>

        <br><br>

        AI-powered road damage detection using
        <strong>YOLOv8n</strong> and the
        <strong>RDD2022</strong> dataset.

        <br><br>

        Computer Vision • Road Safety • Infrastructure Intelligence

    </div>
    """,
    unsafe_allow_html=True,
)
