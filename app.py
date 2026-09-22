from __future__ import annotations

import io
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

from road_damage_detector import detect_road_damage, load_model, model_class_names

st.set_page_config(page_title="RoadGuard", page_icon="🚧", layout="wide")
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "roadguard_best.pt"

st.title("🚧 RoadGuard — Road Damage Detection")
st.caption("Real YOLO inference for visible road damage. Results still require validation against labeled test images.")

with st.sidebar:
    st.header("Detection settings")
    confidence = st.slider("Minimum confidence", 0.05, 0.95, 0.35, 0.05)
    iou = st.slider("IoU / NMS threshold", 0.10, 0.90, 0.45, 0.05)
    image_size = st.selectbox("Inference image size", [320, 416, 512, 640, 768, 1024], index=3)
    enhance = st.checkbox("Improve contrast before inference", value=True)
    st.divider()
    st.write("**Model:**", MODEL_PATH.name)
    st.write("**Classes:** model metadata (not hardcoded labels)")

if not MODEL_PATH.is_file():
    st.error(f"Model file not found: {MODEL_PATH}")
    st.stop()

@st.cache_resource
def cached_model(path: str):
    return load_model(path)

try:
    model = cached_model(str(MODEL_PATH))
    names = model_class_names(model)
    st.success("YOLO model loaded. Predictions below come from model inference.")
    with st.expander("Model classes"):
        st.json(names)
except Exception as exc:
    st.error("Could not load the YOLO model.")
    st.exception(exc)
    st.stop()

uploaded = st.file_uploader("Upload a road image", type=["jpg", "jpeg", "png", "webp", "bmp"])
if uploaded is None:
    st.info("Upload an image to detect cracks, potholes, and other road damage.")
    st.stop()

try:
    image = Image.open(uploaded).convert("RGB")
except Exception as exc:
    st.error("Unable to read this image.")
    st.exception(exc)
    st.stop()

st.image(image, caption=uploaded.name, use_container_width=True)
if not st.button("Analyze road damage", type="primary", use_container_width=True):
    st.stop()

with st.spinner("Running YOLO inference..."):
    try:
        detections = detect_road_damage(
            np.asarray(image), model=model, confidence=confidence, iou=iou,
            image_size=image_size, enhance=enhance,
        )
    except Exception as exc:
        st.error("Inference failed.")
        st.exception(exc)
        st.stop()

annotated = image.copy()
draw = ImageDraw.Draw(annotated)
font = ImageFont.load_default()
for item in detections:
    box = item["bbox"]
    coords = [int(box["x1"]), int(box["y1"]), int(box["x2"]), int(box["y2"])]
    label = f'{item["class_name"]} {item["confidence"]:.2f}'
    draw.rectangle(coords, outline="red", width=4)
    text_box = draw.textbbox((coords[0], coords[1]), label, font=font)
    text_y = max(0, coords[1] - (text_box[3] - text_box[1]) - 6)
    draw.rectangle([coords[0], text_y, coords[0] + (text_box[2] - text_box[0]) + 8, coords[1]], fill="red")
    draw.text((coords[0] + 4, text_y + 3), label, fill="white", font=font)

st.header("Detection result")
left, right = st.columns(2)
with left:
    st.image(image, caption="Original", use_container_width=True)
with right:
    st.image(annotated, caption="Model detections", use_container_width=True)

st.metric("Total detections", len(detections))
if detections:
    st.dataframe([
        {
            "Damage type": item["class_name"],
            "Confidence": f'{item["confidence"]:.2%}',
            "X1": round(item["bbox"]["x1"], 1),
            "Y1": round(item["bbox"]["y1"], 1),
            "X2": round(item["bbox"]["x2"], 1),
            "Y2": round(item["bbox"]["y2"], 1),
        }
        for item in detections
    ], use_container_width=True, hide_index=True)
else:
    st.warning("No object passed the selected confidence threshold. This is not proof that the road is damage-free.")

st.info("Risk/severity is intentionally not presented as ground truth: this repository has no demonstrated validation metrics, and visual bounding boxes cannot measure pothole depth or structural danger.")
report = {
    "project": "RoadGuard",
    "model": MODEL_PATH.name,
    "analysis_time": datetime.now().isoformat(),
    "settings": {"confidence": confidence, "iou": iou, "image_size": image_size, "enhance": enhance},
    "image": {"filename": uploaded.name, "width": image.width, "height": image.height},
    "detections": detections,
}
json_bytes = json.dumps(report, indent=2).encode("utf-8")
st.download_button("Download JSON report", json_bytes, "roadguard_report.json", "application/json")
buffer = io.BytesIO()
annotated.save(buffer, format="PNG")
st.download_button("Download annotated image", buffer.getvalue(), "roadguard_annotated.png", "image/png")
