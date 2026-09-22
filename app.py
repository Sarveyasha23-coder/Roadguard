from __future__ import annotations

import io
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import streamlit as st
from PIL import Image

from inference import (
    CLASS_NAMES,
    DEFAULT_CONFIDENCE,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_IOU,
    calculate_severity,
    detections_to_rows,
    get_model,
    get_model_info,
    predict_image,
    summarize_detections,
)

st.set_page_config(
    page_title="RoadGuard | Infrastructure Intelligence",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "roadguard_best.pt"

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1400px;}
    .roadguard-title {font-size: 2.4rem; font-weight: 800; margin-bottom: .15rem;}
    .roadguard-subtitle {font-size: 1.05rem; opacity: .75; margin-bottom: 1.2rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_model(model_path: str):
    return get_model(model_path)


def image_to_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def make_report(file_name: str, image_size: tuple, detections: List[Dict[str, Any]], summary: Dict[str, Any], severity: Dict[str, Any], settings: Dict[str, Any]) -> bytes:
    report = {
        "project": "RoadGuard Infrastructure Intelligence",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_image": file_name,
        "image": {"width": image_size[0], "height": image_size[1]},
        "settings": settings,
        "summary": summary,
        "severity": severity,
        "detections": detections,
    }
    return json.dumps(report, indent=2, default=str).encode("utf-8")


def sidebar_settings() -> Dict[str, Any]:
    st.sidebar.title("⚙️ Detection Settings")
    st.sidebar.caption("RoadGuard • YOLOv8n • RDD2022")
    confidence = st.sidebar.slider("Confidence threshold", 0.05, 0.95, float(DEFAULT_CONFIDENCE), 0.05)
    iou = st.sidebar.slider("IoU threshold", 0.10, 0.90, float(DEFAULT_IOU), 0.05)
    image_size = st.sidebar.select_slider(
        "Inference image size",
        options=[320, 416, 512, 640, 768, 960, 1280],
        value=DEFAULT_IMAGE_SIZE,
    )
    st.sidebar.divider()
    if MODEL_PATH.exists():
        st.sidebar.success("roadguard_best.pt found")
    else:
        st.sidebar.error("roadguard_best.pt not found")
    st.sidebar.caption(f"Model: {MODEL_PATH.name}")
    return {"confidence": confidence, "iou": iou, "image_size": image_size}


def main() -> None:
    st.markdown('<div class="roadguard-title">🛣️ RoadGuard Infrastructure Intelligence</div>', unsafe_allow_html=True)
    st.markdown('<div class="roadguard-subtitle">AI-powered road damage detection and maintenance prioritization</div>', unsafe_allow_html=True)

    settings = sidebar_settings()

    if not MODEL_PATH.exists():
        st.error("`roadguard_best.pt` was not found. Put it in the same folder as `app.py`.")
        st.code(str(MODEL_PATH), language="text")
        st.stop()

    try:
        with st.spinner("Loading RoadGuard model..."):
            model = load_model(str(MODEL_PATH))
    except Exception as exc:
        st.error("The model could not be loaded.")
        st.exception(exc)
        st.stop()

    with st.expander("Model information", expanded=False):
        info = get_model_info(model)
        a, b, c = st.columns(3)
        a.metric("Task", str(info["task"]))
        b.metric("Model", str(info["model_type"]))
        c.metric("Classes", int(info["num_classes"]))
        mapping = pd.DataFrame([{"ID": k, "RoadGuard Class": v} for k, v in CLASS_NAMES.items()])
        st.dataframe(mapping, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("📷 Upload Road Image")
    uploaded = st.file_uploader("Upload a road image for damage detection", type=["jpg", "jpeg", "png", "webp"])

    if uploaded is None:
        st.info("Upload an image to start detection.")
        st.markdown("### Supported damage classes")
        cols = st.columns(5)
        for col, name in zip(cols, CLASS_NAMES.values()):
            col.info(name)
        st.caption("RoadGuard is a computer-vision decision-support prototype. Severity estimates should not replace professional road inspection.")
        return

    try:
        image = Image.open(uploaded).convert("RGB")
    except Exception as exc:
        st.error("The uploaded file is not a valid image.")
        st.exception(exc)
        return

    width, height = image.size

    if width < 512 or height < 512:
        st.warning(
            f"Uploaded image resolution is **{width} × {height}**. "
            "This is very small for road-damage detection. "
            "Use the original/high-resolution road image rather than a thumbnail or screenshot."
        )

    with st.spinner("Detecting road damage..."):
        try:
            result = predict_image(
                model,
                image,
                confidence=settings["confidence"],
                iou=settings["iou"],
                image_size=settings["image_size"],
            )
        except Exception as exc:
            st.error("Inference failed.")
            st.exception(exc)
            return

    detections = result["detections"]
    candidates = result.get("candidates", [])
    annotated = result["annotated_image"]
    max_raw_confidence = float(result.get("max_raw_confidence", 0.0))
    summary = summarize_detections(detections)
    severity = calculate_severity(detections)

    if not detections:
        if candidates:
            st.warning(
                f"0 final detections at {settings['confidence']:.2f} confidence. "
                f"The model produced {len(candidates)} raw candidate(s) at the internal 0.05 threshold. "
                f"Strongest confidence: {max_raw_confidence:.2%}."
            )
            st.info(
                "Try lowering the Detection confidence slider. "
                "The diagnostic candidates below show what the model actually produced."
            )
        else:
            st.error(
                "The model produced no candidate boxes even at the internal 0.05 threshold. "
                "That points to the image/model rather than the confidence slider."
            )
    else:
        st.success(
            f"Analysis completed — {len(detections)} detection(s) found."
        )

    st.subheader("📊 Inspection Summary")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Detections", summary["total_detections"])
    m2.metric("Average Confidence", f"{summary['average_confidence']:.1%}" if detections else "0.0%")
    m3.metric("Max Raw Confidence", f"{max_raw_confidence:.1%}")
    m4.metric("Image Resolution", f"{width}×{height}")

    st.subheader("🔍 Detection Result")
    left, right = st.columns(2)
    with left:
        st.markdown("**Original image**")
        st.image(image, use_container_width=True)
    with right:
        st.markdown("**RoadGuard detection**")
        st.image(annotated, use_container_width=True)

    st.subheader("🚦 Risk & Maintenance Assessment")
    s1, s2, s3 = st.columns(3)
    s1.metric("Severity Level", severity["severity"])
    s2.metric("Risk Level", severity["risk"])
    s3.metric("Priority", severity["priority"])
    st.caption(f"Severity score: {severity['score']} • Weighted damage score: {severity['weighted_score']}")
    st.info(severity["explanation"])

    st.subheader("📈 Damage Distribution")
    chart_df = pd.DataFrame({"Damage Type": list(summary["counts"].keys()), "Detections": list(summary["counts"].values())})
    if int(chart_df["Detections"].sum()) > 0:
        st.bar_chart(chart_df.set_index("Damage Type"), use_container_width=True)
    else:
        st.info("No damage was detected above the selected confidence threshold.")

    st.subheader("📋 Detection Details")
    rows = detections_to_rows(detections)
    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Confidence": st.column_config.ProgressColumn("Confidence", min_value=0.0, max_value=1.0, format="%.1f%%"),
                "Area": st.column_config.NumberColumn("Box Area", format="%.0f"),
            },
        )
    else:
        st.warning("No road damage was detected. You can try a lower confidence threshold.")

    if candidates:
        with st.expander("🔬 Diagnostic candidates", expanded=False):
            st.caption(
                "Raw candidates returned at the internal 0.05 threshold. "
                "This helps distinguish weak confidence from a true no-prediction case."
            )
            candidate_rows = [
                {
                    "Damage Type": d["class_name"],
                    "Confidence": d["confidence"],
                    "X1": round(d["x1"], 1),
                    "Y1": round(d["y1"], 1),
                    "X2": round(d["x2"], 1),
                    "Y2": round(d["y2"], 1),
                }
                for d in candidates[:20]
            ]
            st.dataframe(
                pd.DataFrame(candidate_rows),
                use_container_width=True,
                hide_index=True,
            )

    with st.expander("Class statistics"):
        stats = pd.DataFrame([
            {"Damage Type": name, "Count": d["count"], "Average Confidence": d["average_confidence"]}
            for name, d in summary["class_details"].items()
        ])
        st.dataframe(
            stats,
            use_container_width=True,
            hide_index=True,
            column_config={"Average Confidence": st.column_config.ProgressColumn("Average Confidence", min_value=0.0, max_value=1.0, format="%.1f%%")},
        )

    st.subheader("⬇️ Export Inspection")
    report = make_report(uploaded.name, image.size, detections, summary, severity, settings)
    annotated_bytes = image_to_bytes(Image.fromarray(annotated))
    d1, d2 = st.columns(2)
    d1.download_button("Download JSON Report", report, "roadguard_report.json", "application/json", use_container_width=True)
    d2.download_button("Download Annotated Image", annotated_bytes, "roadguard_annotated.png", "image/png", use_container_width=True)

    with st.expander("Inference configuration"):
        st.json({
            "confidence_threshold": settings["confidence"],
            "iou_threshold": settings["iou"],
            "image_size": settings["image_size"],
            "model_file": MODEL_PATH.name,
            "source_image": uploaded.name,
            "detections": len(detections),
            "raw_candidates": len(candidates),
            "max_raw_confidence": max_raw_confidence,
            "diagnostic_confidence": 0.05,
        })

    st.divider()
    st.caption("RoadGuard Infrastructure Intelligence • YOLOv8n • RDD2022 • Streamlit")


if __name__ == "__main__":
    main()
