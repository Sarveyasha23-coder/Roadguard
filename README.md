# 🚧 RoadGuard Infrastructure Intelligence

### AI-Powered Road Damage Detection, Severity Assessment & Infrastructure Risk Analysis

RoadGuard Infrastructure Intelligence is an end-to-end computer vision system designed to automatically detect road surface damage from images and transform those detections into actionable infrastructure insights.

The system uses a lightweight **YOLOv8n object detection model** trained on the **RDD2022 (Road Damage Dataset 2022)** dataset and provides an interactive **Streamlit dashboard** for road damage detection, severity estimation, risk assessment, maintenance priority, and visual reporting.

---

## 📌 Project Overview

Poor road infrastructure can create serious safety risks, increase vehicle maintenance costs, and reduce transportation efficiency.

Traditional road inspection methods often depend on:

- Manual inspection
- Human reporting
- Periodic surveys
- Specialized inspection vehicles
- Time-consuming documentation

These approaches can be expensive, slow, and difficult to scale.

**RoadGuard Infrastructure Intelligence** provides an AI-assisted alternative.

A road image is passed through a trained YOLOv8 model. The model identifies visible road defects and returns:

- Damage category
- Bounding box
- Confidence score
- Estimated severity
- Infrastructure risk level
- Maintenance priority
- Recommended action

The final results are presented through an interactive Streamlit interface.

---

# 🎯 Objectives

The main objectives of RoadGuard are:

1. Detect road damage automatically from images.
2. Identify different types of road defects.
3. Provide confidence scores for detections.
4. Estimate the severity of detected damage.
5. Calculate an infrastructure risk level.
6. Assign a maintenance priority.
7. Generate maintenance recommendations.
8. Provide annotated road images.
9. Display detection statistics through a dashboard.
10. Provide downloadable analysis reports.
11. Keep the deployed model lightweight enough for GitHub and practical deployment.

---

# 🧠 Key Features

## 🔍 1. Automatic Road Damage Detection

The system uses YOLOv8n to detect road defects from uploaded images.

The supported damage categories are:

| Class ID | Damage Type |
|---:|---|
| 0 | Longitudinal Crack |
| 1 | Transverse Crack |
| 2 | Alligator Crack |
| 3 | Other Corruption |
| 4 | Pothole |

---

## 📊 2. Confidence-Based Detection

Every prediction contains a confidence score.

Example:

```text
Pothole
Confidence: 91.4%
