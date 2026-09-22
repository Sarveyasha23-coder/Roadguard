```python
from io import BytesIO

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from inference import analyze_image
from severity import analyze_severity


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="RoadGuard",
    page_icon="🚧",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# STYLING
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 0;
    }

    .subtitle {
        font-size: 18px;
        color: #666666;
        margin-top: 0;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🚧 RoadGuard</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    "Infrastructure Intelligence & "
    "Road Damage Detection"
    "</div>",
    unsafe_allow_html=True,
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Detection Settings")

    confidence = st.slider(
        "Confidence Threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.25,
        step=0.05,
    )

    st.divider()

    st.subheader(
        "Supported Damage"
    )

    st.markdown(
        """
        - 🛣️ Longitudinal Crack
        - ↔️ Transverse Crack
        - 🕸️ Alligator Crack
        - ⚠️ Other Corruption
        - 🕳️ Pothole
        """
    )

    st.divider()

    st.caption(
        "RoadGuard uses YOLOv8 for "
        "road-damage detection."
    )


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def convert_to_bgr(
    image: Image.Image,
) -> np.ndarray:

    rgb_image = np.array(
        image.convert("RGB")
    )

    return cv2.cvtColor(
        rgb_image,
        cv2.COLOR_RGB2BGR,
    )


def convert_to_rgb(
    image: np.ndarray,
) -> np.ndarray:

    return cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB,
    )


def create_csv_report(
    detections,
) -> bytes:

    if not detections:

        dataframe = pd.DataFrame(
            [
                {
                    "Damage Type": (
                        "No Damage Detected"
                    ),
                    "Confidence (%)": "",
                    "Severity": "",
                    "Risk Level": "",
                    "Maintenance Priority": "",
                    "Recommended Action": "",
                }
            ]
        )

    else:

        rows = []

        for detection in detections:

            rows.append(
                {
                    "Damage Type": (
                        detection[
                            "damage_type"
                        ]
                    ),
                    "Confidence (%)": (
                        detection[
                            "confidence_percentage"
                        ]
                    ),
                    "Severity Score": (
                        detection[
                            "severity_score"
                        ]
                    ),
                    "Severity": (
                        detection[
                            "severity"
                        ]
                    ),
                    "Risk Score": (
                        detection[
                            "risk_score"
                        ]
                    ),
                    "Risk Level": (
                        detection[
                            "risk_level"
                        ]
                    ),
                    "Maintenance Priority": (
                        detection[
                            "maintenance_priority"
                        ]
                    ),
                    "Recommended Action": (
                        detection[
                            "recommended_action"
                        ]
                    ),
                    "X1": detection["x1"],
                    "Y1": detection["y1"],
                    "X2": detection["x2"],
                    "Y2": detection["y2"],
                    "Area (%)": (
                        detection[
                            "area_percentage"
                        ]
                    ),
                }
            )

        dataframe = pd.DataFrame(
            rows
        )

    return dataframe.to_csv(
        index=False
    ).encode("utf-8")


# ============================================================
# IMAGE UPLOAD
# ============================================================

st.subheader(
    "📤 Upload Road Image"
)

uploaded_file = st.file_uploader(
    "Choose a road image",
    type=[
        "jpg",
        "jpeg",
        "png",
        "webp",
    ],
)


# ============================================================
# NO IMAGE
# ============================================================

if uploaded_file is None:

    st.info(
        "👆 Upload a road image to begin analysis."
    )

    st.markdown(
        """
        ### How RoadGuard Works

        **1. Upload** a road image.

        **2. Detect** road defects using YOLOv8.

        **3. Analyze** severity and infrastructure risk.

        **4. Prioritize** maintenance.

        **5. Download** the analysis report.
        """
    )

    st.stop()


# ============================================================
# LOAD IMAGE
# ============================================================

try:

    original_image = Image.open(
        uploaded_file
    ).convert("RGB")

except Exception as error:

    st.error(
        f"Unable to read image: {error}"
    )

    st.stop()


# ============================================================
# DISPLAY ORIGINAL IMAGE
# ============================================================

col1, col2 = st.columns(2)

with col1:

    st.subheader(
        "📷 Original Image"
    )

    st.image(
        original_image,
        use_container_width=True,
    )


# ============================================================
# RUN YOLO
# ============================================================

image_bgr = convert_to_bgr(
    original_image
)

with st.spinner(
    "🔍 Detecting road damage..."
):

    try:

        inference_result = analyze_image(
            image=image_bgr,
            confidence=confidence,
        )

    except Exception as error:

        st.error(
            "Model inference failed."
        )

        st.exception(error)

        st.stop()


detections = inference_result[
    "detections"
]


# ============================================================
# SEVERITY ANALYSIS
# ============================================================

severity_result = analyze_severity(
    detections
)

analyzed_detections = (
    severity_result[
        "detections"
    ]
)

annotated_image = (
    inference_result[
        "annotated_image"
    ]
)


# ============================================================
# DISPLAY RESULT
# ============================================================

with col2:

    st.subheader(
        "🎯 Detection Result"
    )

    st.image(
        convert_to_rgb(
            annotated_image
        ),
        use_container_width=True,
    )


# ============================================================
# SUMMARY
# ============================================================

st.divider()

st.subheader(
    "📊 Analysis Summary"
)

metric1, metric2, metric3, metric4 = (
    st.columns(4)
)

with metric1:

    st.metric(
        "Detected Defects",
        inference_result[
            "detection_count"
        ],
    )

with metric2:

    st.metric(
        "Average Confidence",
        f"{inference_result['average_confidence']:.1f}%",
    )

with metric3:

    st.metric(
        "Overall Severity",
        severity_result[
            "overall_severity"
        ],
    )

with metric4:

    st.metric(
        "Risk Score",
        f"{severity_result['overall_risk_score']:.1f}/100",
    )


# ============================================================
# INFRASTRUCTURE RISK
# ============================================================

st.subheader(
    "🏗️ Infrastructure Risk"
)

risk1, risk2, risk3 = st.columns(3)

with risk1:

    st.metric(
        "Risk Level",
        severity_result[
            "overall_risk"
        ],
    )

with risk2:

    st.metric(
        "Maintenance Priority",
        severity_result[
            "maintenance_priority"
        ],
    )

with risk3:

    st.metric(
        "Risk Score",
        f"{severity_result['overall_risk_score']:.1f}/100",
    )


st.info(
    "💡 "
    + severity_result[
        "overall_recommendation"
    ]
)


# ============================================================
# DETECTION DETAILS
# ============================================================

st.divider()

st.subheader(
    "🔎 Detection Details"
)

if not analyzed_detections:

    st.success(
        "No road damage was detected above "
        f"the {confidence:.0%} threshold."
    )

else:

    rows = []

    for index, detection in enumerate(
        analyzed_detections,
        start=1,
    ):

        rows.append(
            {
                "#": index,
                "Damage Type": (
                    detection[
                        "damage_type"
                    ]
                ),
                "Confidence": (
                    f"{detection['confidence_percentage']:.1f}%"
                ),
                "Severity": (
                    detection[
                        "severity"
                    ]
                ),
                "Risk": (
                    detection[
                        "risk_level"
                    ]
                ),
                "Priority": (
                    detection[
                        "maintenance_priority"
                    ]
                ),
                "Area": (
                    f"{detection['area_percentage']:.2f}%"
                ),
            }
        )

    dataframe = pd.DataFrame(
        rows
    )

    st.dataframe(
        dataframe,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# MAINTENANCE RECOMMENDATIONS
# ============================================================

if analyzed_detections:

    st.subheader(
        "🛠️ Recommended Maintenance Actions"
    )

    for index, detection in enumerate(
        analyzed_detections,
        start=1,
    ):

        with st.expander(
            f"{index}. "
            f"{detection['damage_type']} — "
            f"{detection['severity']}"
        ):

            col_a, col_b = st.columns(2)

            with col_a:

                st.write(
                    "**Confidence:** "
                    f"{detection['confidence_percentage']:.1f}%"
                )

                st.write(
                    "**Severity Score:** "
                    f"{detection['severity_score']:.1f}/100"
                )

                st.write(
                    "**Severity:** "
                    f"{detection['severity']}"
                )

                st.write(
                    "**Risk Score:** "
                    f"{detection['risk_score']:.1f}/100"
                )

            with col_b:

                st.write(
                    "**Risk Level:** "
                    f"{detection['risk_level']}"
                )

                st.write(
                    "**Maintenance Priority:** "
                    f"{detection['maintenance_priority']}"
                )

                st.write(
                    "**Affected Area:** "
                    f"{detection['area_percentage']:.2f}%"
                )

            st.markdown(
                "**Recommended Action**"
            )

            st.info(
                detection[
                    "recommended_action"
                ]
            )


# ============================================================
# DETECTION STATISTICS
# ============================================================

if detections:

    st.divider()

    st.subheader(
        "📈 Detection Statistics"
    )

    statistics = (
        inference_result[
            "class_counts"
        ]
    )

    columns = st.columns(
        max(
            1,
            min(
                len(statistics),
                4,
            ),
        )
    )

    for index, (
        damage_type,
        count,
    ) in enumerate(
        statistics.items()
    ):

        with columns[
            index % len(columns)
        ]:

            st.metric(
                damage_type,
                count,
            )


# ============================================================
# DOWNLOAD CSV REPORT
# ============================================================

st.divider()

st.subheader(
    "📥 Download Analysis"
)

report_data = create_csv_report(
    analyzed_detections
)

st.download_button(
    label="⬇️ Download CSV Report",
    data=report_data,
    file_name="roadguard_analysis.csv",
    mime="text/csv",
)


# ============================================================
# DOWNLOAD ANNOTATED IMAGE
# ============================================================

annotated_rgb = convert_to_rgb(
    annotated_image
)

annotated_pil = Image.fromarray(
    annotated_rgb
)

image_buffer = BytesIO()

annotated_pil.save(
    image_buffer,
    format="PNG",
)

st.download_button(
    label="🖼️ Download Annotated Image",
    data=image_buffer.getvalue(),
    file_name="roadguard_annotated.png",
    mime="image/png",
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "RoadGuard Infrastructure Intelligence • "
    "AI-assisted road damage analysis"
)

st.caption(
    "Severity and risk values are heuristic estimates "
    "based on model detections and should be validated "
    "with field inspection before operational decisions."
)
```
