"""
RoadGuard Infrastructure Intelligence
=====================================

Streamlit application for road-damage detection using a YOLOv8
model trained on the RDD2022 dataset.

Expected project structure:

Roadguard/
├── app.py
├── roadguard_best.pt
├── requirements.txt
└── README.md

The app is intentionally self-contained: it does not require
inference.py or severity.py to run.
"""

from __future__ import annotations

import io
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="RoadGuard Infrastructure Intelligence",
    page_icon="🚧",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PATHS AND CONSTANTS
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

CLASS_WEIGHTS = {
    "Longitudinal Crack": 25,
    "Transverse Crack": 25,
    "Alligator Crack": 35,
    "Other Corruption": 20,
    "Pothole": 40,
}


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
<style>
.roadguard-header {
    padding: 28px;
    border-radius: 18px;
    margin-bottom: 24px;
    color: white;
    background: linear-gradient(135deg, #111827, #374151);
}

.roadguard-title {
    font-size: 36px;
    font-weight: 800;
    margin-bottom: 8px;
}

.roadguard-subtitle {
    font-size: 16px;
    line-height: 1.6;
    opacity: 0.92;
}

.metric-card {
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 14px;
    padding: 18px;
    min-height: 115px;
    box-shadow: 0 3px 12px rgba(0, 0, 0, 0.05);
}

.metric-title {
    color: #6b7280;
    font-size: 13px;
    font-weight: 600;
}

.metric-value {
    color: #111827;
    font-size: 28px;
    font-weight: 800;
    margin-top: 6px;
}

.metric-description {
    color: #6b7280;
    font-size: 12px;
    margin-top: 4px;
}

.info-box {
    background: #eff6ff;
    border-left: 5px solid #2563eb;
    padding: 15px;
    border-radius: 9px;
    margin: 12px 0;
}

.warning-box {
    background: #fffbeb;
    border-left: 5px solid #d97706;
    padding: 15px;
    border-radius: 9px;
    margin: 12px 0;
}

.footer {
    text-align: center;
    color: #6b7280;
    font-size: 13px;
    border-top: 1px solid #e5e7eb;
    margin-top: 45px;
    padding: 20px;
}
</style>
""",
    unsafe_allow_html=True,
)


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
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_model(model_path: str) -> YOLO:
    """Load and cache the YOLO model."""
    return YOLO(model_path)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("⚙️ RoadGuard Controls")

    confidence_threshold = st.slider(
        "Detection confidence",
        min_value=0.05,
        max_value=0.95,
        value=0.35,
        step=0.05,
    )

    iou_threshold = st.slider(
        "IoU threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.45,
        step=0.05,
    )

    image_size = st.selectbox(
        "Inference image size",
        [320, 416, 512, 640, 768],
        index=3,
    )

    st.divider()

    st.subheader("📊 Project")

    st.write("**Dataset:** RDD2022")
    st.write("**Model:** YOLOv8n")
    st.write("**Classes:** 5")
    st.write("**Framework:** Ultralytics + Streamlit")

    st.divider()

    st.subheader("📁 Model")

    st.code(str(MODEL_PATH), language="text")


# ============================================================
# CHECK MODEL FILE
# ============================================================

if not MODEL_PATH.exists():
    st.error("❌ Model file not found.")

    st.warning(
        f"""
Expected file:

{MODEL_PATH}

Make sure `roadguard_best.pt` is in the same folder as `app.py`.
"""
    )

    st.stop()


# ============================================================
# LOAD MODEL
# ============================================================

try:
    model = load_model(str(MODEL_PATH))
except Exception as exc:
    st.error("❌ Could not load the YOLO model.")
    st.exception(exc)
    st.stop()


st.success("✅ RoadGuard YOLO model loaded successfully.")


# ============================================================
# MODEL INFORMATION
# ============================================================

with st.expander("🤖 Model Information"):
    model_names = getattr(model, "names", None)

    st.json(
        {
            "model_file": MODEL_PATH.name,
            "model_type": "YOLOv8n",
            "dataset": "RDD2022",
            "classes": model_names if model_names else CLASS_NAMES,
        }
    )


# ============================================================
# UPLOAD IMAGE
# ============================================================

st.header("📷 Road Image Analysis")

uploaded_file = st.file_uploader(
    "Upload a road image",
    type=["jpg", "jpeg", "png", "webp", "bmp"],
)

if uploaded_file is None:
    st.info(
        "👆 Upload a road image and click **Analyze Road Damage**."
    )

    st.subheader("🧠 Detection Classes")

    columns = st.columns(5)

    for index, class_name in CLASS_NAMES.items():
        with columns[index]:
            st.markdown(
                f"""
<div class="metric-card">
    <div class="metric-title">Class {index}</div>
    <div style="font-weight:700;margin-top:8px;">
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
# READ IMAGE
# ============================================================

try:
    original_image = Image.open(uploaded_file).convert("RGB")
except Exception as exc:
    st.error("❌ Unable to read the uploaded image.")
    st.exception(exc)
    st.stop()

image_array = np.asarray(original_image)
image_width, image_height = original_image.size


# ============================================================
# IMAGE PREVIEW
# ============================================================

with st.expander("🖼️ Uploaded Image", expanded=True):
    st.image(
        original_image,
        caption=uploaded_file.name,
        use_container_width=True,
    )


# ============================================================
# ANALYZE BUTTON
# ============================================================

analyze_button = st.button(
    "🚀 Analyze Road Damage",
    type="primary",
    use_container_width=True,
)


# ============================================================
# RUN INFERENCE
# ============================================================

if analyze_button:
    with st.spinner("Running RoadGuard AI analysis..."):
        try:
            results = model.predict(
                source=image_array,
                conf=confidence_threshold,
                iou=iou_threshold,
                imgsz=image_size,
                verbose=False,
            )
        except Exception as exc:
            st.error("❌ RoadGuard inference failed.")
            st.exception(exc)
            st.stop()

        result = results[0]

        detections: list[dict[str, Any]] = []

        if result.boxes is not None:
            boxes = result.boxes

            xyxy = boxes.xyxy.cpu().numpy()
            confidences = boxes.conf.cpu().numpy()
            class_ids = boxes.cls.cpu().numpy().astype(int)

            for box, confidence, class_id in zip(
                xyxy,
                confidences,
                class_ids,
            ):
                x1, y1, x2, y2 = [float(value) for value in box]

                width = max(0.0, x2 - x1)
                height = max(0.0, y2 - y1)
                area = width * height

                # Prefer the names stored inside the trained model.
                model_names = getattr(model, "names", None)

                if model_names:
                    class_name = str(
                        model_names.get(
                            int(class_id),
                            CLASS_NAMES.get(
                                int(class_id),
                                f"Class {class_id}",
                            ),
                        )
                    )
                else:
                    class_name = CLASS_NAMES.get(
                        int(class_id),
                        f"Class {class_id}",
                    )

                detections.append(
                    {
                        "class_id": int(class_id),
                        "class_name": class_name,
                        "confidence": float(confidence),
                        "bbox": {
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2,
                        },
                        "area": area,
                    }
                )

        st.session_state["roadguard_detections"] = detections
        st.session_state["roadguard_image"] = original_image
        st.session_state["roadguard_filename"] = uploaded_file.name
        st.session_state["roadguard_analysis_time"] = (
            datetime.now().isoformat()
        )

        st.success(
            f"✅ Analysis completed — {len(detections)} detection(s) found."
        )


# ============================================================
# CHECK FOR RESULTS
# ============================================================

if "roadguard_detections" not in st.session_state:
    st.info("Click **Analyze Road Damage** to run the model.")
    st.stop()


# ============================================================
# GET RESULTS
# ============================================================

detections = st.session_state["roadguard_detections"]
analysis_image = st.session_state["roadguard_image"]
filename = st.session_state.get(
    "roadguard_filename",
    "road_image.jpg",
)


# ============================================================
# SUMMARY VALUES
# ============================================================

confidence_values = [
    float(item["confidence"])
    for item in detections
]

total_detections = len(detections)

average_confidence = (
    float(np.mean(confidence_values))
    if confidence_values
    else 0.0
)

highest_confidence = (
    float(np.max(confidence_values))
    if confidence_values
    else 0.0
)


# ============================================================
# TOP METRICS
# ============================================================

st.header("📊 Detection Overview")

metric_columns = st.columns(4)

with metric_columns[0]:
    st.markdown(
        f"""
<div class="metric-card">
    <div class="metric-title">Total Detections</div>
    <div class="metric-value">{total_detections}</div>
    <div class="metric-description">Detected road-damage objects</div>
</div>
""",
        unsafe_allow_html=True,
    )

with metric_columns[1]:
    st.markdown(
        f"""
<div class="metric-card">
    <div class="metric-title">Average Confidence</div>
    <div class="metric-value">{average_confidence:.1%}</div>
    <div class="metric-description">Mean model confidence</div>
</div>
""",
        unsafe_allow_html=True,
    )

with metric_columns[2]:
    st.markdown(
        f"""
<div class="metric-card">
    <div class="metric-title">Highest Confidence</div>
    <div class="metric-value">{highest_confidence:.1%}</div>
    <div class="metric-description">Strongest detection</div>
</div>
""",
        unsafe_allow_html=True,
    )

with metric_columns[3]:
    st.markdown(
        f"""
<div class="metric-card">
    <div class="metric-title">Image Resolution</div>
    <div class="metric-value">{image_width}×{image_height}</div>
    <div class="metric-description">Uploaded image size</div>
</div>
""",
        unsafe_allow_html=True,
    )


# ============================================================
# ANNOTATED IMAGE
# ============================================================

annotated_image = analysis_image.copy()
draw = ImageDraw.Draw(annotated_image)

try:
    font = ImageFont.load_default()
except Exception:
    font = None


for detection in detections:
    bbox = detection["bbox"]

    x1 = int(bbox["x1"])
    y1 = int(bbox["y1"])
    x2 = int(bbox["x2"])
    y2 = int(bbox["y2"])

    class_name = detection["class_name"]
    confidence = float(detection["confidence"])

    label = f"{class_name} {confidence:.2f}"

    draw.rectangle(
        [x1, y1, x2, y2],
        outline="red",
        width=4,
    )

    try:
        text_box = draw.textbbox(
            (x1, y1),
            label,
            font=font,
        )
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]
    except Exception:
        text_width = len(label) * 7
        text_height = 14

    label_y = max(0, y1 - text_height - 6)

    draw.rectangle(
        [
            x1,
            label_y,
            x1 + text_width + 8,
            label_y + text_height + 6,
        ],
        fill="red",
    )

    draw.text(
        [x1 + 4, label_y + 3],
        label,
        fill="white",
        font=font,
    )


# ============================================================
# IMAGE RESULTS
# ============================================================

st.header("🖼️ Detection Result")

image_col1, image_col2 = st.columns(2)

with image_col1:
    st.subheader("Original Image")
    st.image(
        analysis_image,
        use_container_width=True,
    )

with image_col2:
    st.subheader("AI Detection")
    st.image(
        annotated_image,
        use_container_width=True,
    )


# ============================================================
# SEVERITY CALCULATION
# ============================================================

def calculate_severity(
    detections_list: list[dict[str, Any]],
    width: int,
    height: int,
) -> dict[str, Any]:
    """
    Calculate an application-level visual risk indicator.

    This is not a structural engineering diagnosis.
    """

    if not detections_list:
        return {
            "severity": "Low",
            "risk_score": 0,
            "priority": "Routine Inspection",
            "recommendation": (
                "No road damage was detected above the "
                "selected confidence threshold."
            ),
        }

    image_area = max(1, width * height)
    weighted_scores = []

    for detection in detections_list:
        class_name = detection["class_name"]
        confidence = float(detection["confidence"])
        area = float(detection["area"])

        # Match known names first. If a custom model name is used,
        # use a conservative default weight.
        base_weight = CLASS_WEIGHTS.get(class_name, 20)

        confidence_score = base_weight * confidence

        area_ratio = area / image_area
        area_bonus = min(area_ratio * 100.0, 20.0)

        weighted_scores.append(
            confidence_score + area_bonus
        )

    raw_score = sum(weighted_scores)

    count_bonus = min(
        len(detections_list) * 5,
        25,
    )

    risk_score = min(
        100,
        int(raw_score + count_bonus),
    )

    if risk_score >= 75:
        severity = "Critical"
        priority = "Immediate Intervention"
        recommendation = (
            "High-impact visual road-damage indicators were detected. "
            "Prompt field inspection and maintenance planning are recommended."
        )
    elif risk_score >= 50:
        severity = "High"
        priority = "High Priority"
        recommendation = (
            "Significant visual road-damage indicators were detected. "
            "A field inspection and maintenance assessment should be scheduled."
        )
    elif risk_score >= 25:
        severity = "Medium"
        priority = "Planned Maintenance"
        recommendation = (
            "Moderate visual road-damage indicators were detected. "
            "The location should be monitored and considered for planned maintenance."
        )
    else:
        severity = "Low"
        priority = "Routine Inspection"
        recommendation = (
            "Limited visual road-damage indicators were detected. "
            "Routine monitoring is recommended."
        )

    return {
        "severity": severity,
        "risk_score": risk_score,
        "priority": priority,
        "recommendation": recommendation,
    }


severity_result = calculate_severity(
    detections,
    image_width,
    image_height,
)

severity_level = severity_result["severity"]
risk_score = int(severity_result["risk_score"])
priority = severity_result["priority"]
recommendation = severity_result["recommendation"]


# ============================================================
# RISK ASSESSMENT
# ============================================================

st.header("🚦 Infrastructure Risk Assessment")

st.metric(
    "Risk Score",
    f"{risk_score}/100",
)

risk_col1, risk_col2 = st.columns(2)

with risk_col1:
    st.metric("Severity", severity_level)

with risk_col2:
    st.metric("Maintenance Priority", priority)

if severity_level == "Critical":
    st.error(
        f"**{severity_level} Severity** — {recommendation}"
    )
elif severity_level == "High":
    st.warning(
        f"**{severity_level} Severity** — {recommendation}"
    )
elif severity_level == "Medium":
    st.info(
        f"**{severity_level} Severity** — {recommendation}"
    )
else:
    st.success(
        f"**{severity_level} Severity** — {recommendation}"
    )


# ============================================================
# DETECTION TABLE
# ============================================================

st.header("🔎 Detailed Detections")

if detections:
    table_rows = []

    for index, detection in enumerate(
        detections,
        start=1,
    ):
        bbox = detection["bbox"]

        table_rows.append(
            {
                "Detection": index,
                "Damage Type": detection["class_name"],
                "Confidence": (
                    f'{float(detection["confidence"]):.2%}'
                ),
                "X1": round(float(bbox["x1"]), 1),
                "Y1": round(float(bbox["y1"]), 1),
                "X2": round(float(bbox["x2"]), 1),
                "Y2": round(float(bbox["y2"]), 1),
                "Area": round(float(detection["area"]), 1),
            }
        )

    detection_dataframe = pd.DataFrame(table_rows)

    st.dataframe(
        detection_dataframe,
        use_container_width=True,
        hide_index=True,
    )
else:
    st.info(
        "No road damage was detected above the selected confidence threshold."
    )


# ============================================================
# DAMAGE DISTRIBUTION
# ============================================================

st.header("📈 Damage Distribution")

if detections:
    class_counts: dict[str, int] = {}

    for detection in detections:
        name = detection["class_name"]
        class_counts[name] = class_counts.get(name, 0) + 1

    chart_dataframe = pd.DataFrame(
        {
            "Damage Type": list(class_counts.keys()),
            "Detections": list(class_counts.values()),
        }
    ).set_index("Damage Type")

    st.bar_chart(chart_dataframe)
else:
    st.info("No detection distribution is available.")


# ============================================================
# CLASS ANALYSIS
# ============================================================

st.header("🧩 Class Analysis")

class_columns = st.columns(len(CLASS_NAMES))

for index, class_name in CLASS_NAMES.items():
    count = sum(
        1
        for detection in detections
        if int(detection["class_id"]) == index
    )

    with class_columns[index]:
        st.metric(
            class_name,
            count,
        )


# ============================================================
# JSON REPORT
# ============================================================

st.header("📄 Analysis Report")

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


json_data = json.dumps(
    report,
    indent=4,
    ensure_ascii=False,
)


download_col1, download_col2 = st.columns(2)

with download_col1:
    st.download_button(
        "📥 Download JSON Report",
        data=json_data,
        file_name="roadguard_analysis_report.json",
        mime="application/json",
        use_container_width=True,
    )

with download_col2:
    image_buffer = io.BytesIO()

    annotated_image.save(
        image_buffer,
        format="PNG",
    )

    image_buffer.seek(0)

    st.download_button(
        "🖼️ Download Annotated Image",
        data=image_buffer,
        file_name="roadguard_annotated.png",
        mime="image/png",
        use_container_width=True,
    )


# ============================================================
# DISCLAIMER
# ============================================================

st.markdown(
    """
<div class="warning-box">
<strong>⚠️ Important Note</strong><br><br>
RoadGuard provides computer-vision-based visual assessment
and maintenance-prioritization indicators. The displayed
severity and risk score are not a substitute for professional
civil-engineering inspection or structural assessment.
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
    AI-powered road damage detection using YOLOv8n and RDD2022.
    <br><br>
    Computer Vision • Road Safety • Infrastructure Intelligence
</div>
""",
    unsafe_allow_html=True,
)
