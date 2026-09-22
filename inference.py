"""
RoadGuard Infrastructure Intelligence
======================================

Computer Vision inference system for road damage detection.

Model:
    YOLOv8n trained on RDD2022

Supported inputs:
    - Single image
    - Folder of images
    - Video
    - Webcam

Outputs:
    - Annotated images/video
    - Damage detections
    - Confidence scores
    - Estimated severity
    - Risk score
    - Priority level
    - JSON report

IMPORTANT:
    Risk/severity are engineered decision-support scores.
    They are NOT ground-truth labels from RDD2022.

Author: RoadGuard Team
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    print("ERROR: Ultralytics is not installed.")
    print("Run: pip install ultralytics")
    sys.exit(1)


# ============================================================
# CONFIGURATION
# ============================================================

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DEFAULT_MODEL = str(BASE_DIR / "roadguard_best.pt")

# RDD2022 classes used during training.
CLASS_NAMES = {
    0: "Longitudinal Crack",
    1: "Transverse Crack",
    2: "Alligator Crack",
    3: "Other Corruption",
    4: "Pothole",
}


# These are engineered weights for RoadGuard's
# decision-support score. They are NOT learned labels.
DAMAGE_WEIGHTS = {
    "Longitudinal Crack": 0.70,
    "Transverse Crack": 0.65,
    "Alligator Crack": 0.85,
    "Other Corruption": 0.50,
    "Pothole": 1.00,
}


# ============================================================
# ROADGUARD ENGINE
# ============================================================

class RoadGuard:
    """
    Main RoadGuard inference engine.

    Responsibilities:
        1. Load YOLO model
        2. Run object detection
        3. Calculate damage area
        4. Estimate severity
        5. Calculate risk score
        6. Determine priority
        7. Produce annotated output
        8. Generate JSON reports
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL,
        confidence: float = 0.25,
        iou: float = 0.45,
        image_size: int = 640,
        device: Optional[str] = None,
    ):
        self.model_path = Path(model_path)
        self.confidence = confidence
        self.iou = iou
        self.image_size = image_size
        self.device = device

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"\nModel not found:\n{self.model_path}\n\n"
                "Make sure roadguard_best.pt exists in:\n"
                "models/roadguard_best.pt"
            )

        print("=" * 60)
        print("ROADGUARD INITIALIZATION")
        print("=" * 60)
        print(f"Model:      {self.model_path}")
        print(f"Confidence: {self.confidence}")
        print(f"IoU:        {self.iou}")
        print(f"Image size: {self.image_size}")
        print(f"Device:     {self.device or 'Auto'}")
        print("=" * 60)

        self.model = YOLO(str(self.model_path))

        print("Model loaded successfully.\n")

    # --------------------------------------------------------
    # DAMAGE AREA
    # --------------------------------------------------------

    @staticmethod
    def calculate_area_ratio(
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        image_width: int,
        image_height: int,
    ) -> float:
        """
        Calculate the percentage of the image covered
        by a detection bounding box.
        """

        box_width = max(0.0, x2 - x1)
        box_height = max(0.0, y2 - y1)

        box_area = box_width * box_height
        image_area = max(
            1.0,
            float(image_width * image_height)
        )

        return float(box_area / image_area)

    # --------------------------------------------------------
    # SEVERITY
    # --------------------------------------------------------

    @staticmethod
    def calculate_severity(
        area_ratio: float,
        confidence: float,
    ) -> str:
        """
        Estimate severity from visible damage extent
        and model confidence.

        This is an engineered estimate.
        """

        # Convert area ratio into a 0-100 scale.
        #
        # We cap the contribution so a very large
        # detection does not dominate the entire score.
        area_score = min(area_ratio * 100.0, 100.0)

        severity_signal = (
            0.70 * area_score +
            0.30 * (confidence * 100.0)
        )

        if severity_signal >= 60:
            return "HIGH"

        if severity_signal >= 30:
            return "MEDIUM"

        return "LOW"

    # --------------------------------------------------------
    # RISK SCORE
    # --------------------------------------------------------

    @staticmethod
    def calculate_risk_score(
        damage_name: str,
        confidence: float,
        area_ratio: float,
    ) -> float:
        """
        Calculate RoadGuard's engineered 0-100 risk score.

        Components:
            45% damage type
            30% model confidence
            25% visible area

        NOTE:
            This is NOT a medically/statistically validated
            risk probability.
        """

        damage_weight = DAMAGE_WEIGHTS.get(
            damage_name,
            0.50
        )

        damage_score = damage_weight * 100.0

        confidence_score = confidence * 100.0

        area_score = min(
            area_ratio * 100.0,
            100.0
        )

        risk = (
            0.45 * damage_score +
            0.30 * confidence_score +
            0.25 * area_score
        )

        return round(
            float(np.clip(risk, 0.0, 100.0)),
            2
        )

    # --------------------------------------------------------
    # PRIORITY
    # --------------------------------------------------------

    @staticmethod
    def get_priority(risk_score: float) -> str:
        """
        Convert risk score into an operational priority.
        """

        if risk_score >= 70:
            return "HIGH"

        if risk_score >= 40:
            return "MEDIUM"

        return "LOW"

    # --------------------------------------------------------
    # RECOMMENDATION
    # --------------------------------------------------------

    @staticmethod
    def get_recommendation(
        damage_name: str,
        priority: str,
        severity: str,
    ) -> str:
        """
        Generate a human-readable recommendation.
        """

        if priority == "HIGH":
            return (
                "Immediate inspection recommended. "
                "Consider prioritizing this damage for maintenance."
            )

        if priority == "MEDIUM":
            return (
                "Schedule inspection and monitor the "
                "damage condition."
            )

        return (
            "Monitor the detected damage during "
            "future road inspections."
        )

    # --------------------------------------------------------
    # SINGLE IMAGE INFERENCE
    # --------------------------------------------------------

    def predict_image(
        self,
        image_path: str | Path,
    ) -> Dict:

        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(
                f"Input image not found: {image_path}"
            )

        image = cv2.imread(str(image_path))

        if image is None:
            raise ValueError(
                f"Unable to read image: {image_path}"
            )

        height, width = image.shape[:2]

        # Run YOLO
        results = self.model.predict(
            source=str(image_path),
            conf=self.confidence,
            iou=self.iou,
            imgsz=self.image_size,
            device=self.device,
            verbose=False,
        )

        result = results[0]

        detections: List[Dict] = []

        if result.boxes is not None:

            boxes = result.boxes

            for index in range(len(boxes)):

                cls_id = int(
                    boxes.cls[index].item()
                )

                confidence = float(
                    boxes.conf[index].item()
                )

                xyxy = boxes.xyxy[index].cpu().numpy()

                x1, y1, x2, y2 = map(
                    float,
                    xyxy
                )

                damage_name = CLASS_NAMES.get(
                    cls_id,
                    f"Unknown Class {cls_id}"
                )

                area_ratio = self.calculate_area_ratio(
                    x1,
                    y1,
                    x2,
                    y2,
                    width,
                    height,
                )

                severity = self.calculate_severity(
                    area_ratio,
                    confidence,
                )

                risk_score = self.calculate_risk_score(
                    damage_name,
                    confidence,
                    area_ratio,
                )

                priority = self.get_priority(
                    risk_score
                )

                recommendation = self.get_recommendation(
                    damage_name,
                    priority,
                    severity,
                )

                detection = {
                    "id": index + 1,
                    "class_id": cls_id,
                    "damage_type": damage_name,
                    "confidence": round(
                        confidence,
                        4
                    ),
                    "bounding_box": {
                        "x1": round(x1, 2),
                        "y1": round(y1, 2),
                        "x2": round(x2, 2),
                        "y2": round(y2, 2),
                    },
                    "area_ratio": round(
                        area_ratio,
                        5
                    ),
                    "area_percentage": round(
                        area_ratio * 100,
                        3
                    ),
                    "severity": severity,
                    "risk_score": risk_score,
                    "priority": priority,
                    "recommendation": recommendation,
                }

                detections.append(detection)

        # ----------------------------------------------------
        # IMAGE-LEVEL SUMMARY
        # ----------------------------------------------------

        if detections:

            highest_risk = max(
                detections,
                key=lambda x: x["risk_score"]
            )

            overall_risk = max(
                x["risk_score"]
                for x in detections
            )

            overall_priority = self.get_priority(
                overall_risk
            )

            damage_types = sorted(
                set(
                    x["damage_type"]
                    for x in detections
                )
            )

        else:

            highest_risk = None
            overall_risk = 0.0
            overall_priority = "LOW"
            damage_types = []

        report = {
            "system": "RoadGuard Infrastructure Intelligence",
            "timestamp": datetime.now().isoformat(),
            "image": str(image_path),
            "image_width": width,
            "image_height": height,
            "detections_count": len(detections),
            "damage_types_detected": damage_types,
            "overall_risk_score": round(
                overall_risk,
                2
            ),
            "overall_priority": overall_priority,
            "highest_risk_detection": highest_risk,
            "detections": detections,
        }

        return {
            "report": report,
            "result": result,
            "original_image": image,
        }

    # --------------------------------------------------------
    # DRAW PROFESSIONAL ROADGUARD OUTPUT
    # --------------------------------------------------------

    def draw_results(
        self,
        inference_output: Dict,
    ) -> np.ndarray:

        report = inference_output["report"]
        image = inference_output[
            "original_image"
        ].copy()

        detections = report["detections"]

        # -----------------------------------------------
        # Draw bounding boxes
        # -----------------------------------------------

        for detection in detections:

            box = detection["bounding_box"]

            x1 = int(box["x1"])
            y1 = int(box["y1"])
            x2 = int(box["x2"])
            y2 = int(box["y2"])

            risk = detection["risk_score"]

            # Color based on priority
            if detection["priority"] == "HIGH":
                color = (0, 0, 255)

            elif detection["priority"] == "MEDIUM":
                color = (0, 165, 255)

            else:
                color = (0, 200, 0)

            # Bounding box
            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                color,
                3,
            )

            # Label
            label = (
                f"{detection['damage_type']} | "
                f"{detection['confidence'] * 100:.0f}% | "
                f"Risk {risk:.0f}"
            )

            font = cv2.FONT_HERSHEY_SIMPLEX

            font_scale = 0.55
            thickness = 2

            (text_width, text_height), baseline = (
                cv2.getTextSize(
                    label,
                    font,
                    font_scale,
                    thickness,
                )
            )

            label_y = max(
                text_height + 10,
                y1
            )

            # Background
            cv2.rectangle(
                image,
                (
                    x1,
                    label_y - text_height - 10,
                ),
                (
                    x1 + text_width + 10,
                    label_y + baseline - 5,
                ),
                color,
                -1,
            )

            # Text
            cv2.putText(
                image,
                label,
                (
                    x1 + 5,
                    label_y - 5,
                ),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA,
            )

        # -----------------------------------------------
        # Dashboard panel
        # -----------------------------------------------

        panel_width = 420

        panel = np.zeros(
            (
                image.shape[0],
                panel_width,
                3,
            ),
            dtype=np.uint8,
        )

        # Dark panel
        panel[:] = (30, 30, 30)

        y = 45

        font = cv2.FONT_HERSHEY_SIMPLEX

        # Title
        cv2.putText(
            panel,
            "ROADGUARD",
            (25, y),
            font,
            1.1,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )

        y += 45

        cv2.putText(
            panel,
            "Infrastructure Intelligence",
            (25, y),
            font,
            0.55,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

        y += 45

        # Summary
        cv2.putText(
            panel,
            f"Damage detected: {report['detections_count']}",
            (25, y),
            font,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        y += 38

        cv2.putText(
            panel,
            f"Risk score: {report['overall_risk_score']:.0f}/100",
            (25, y),
            font,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        y += 38

        cv2.putText(
            panel,
            f"Priority: {report['overall_priority']}",
            (25, y),
            font,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        y += 50

        cv2.line(
            panel,
            (25, y),
            (panel_width - 25, y),
            (100, 100, 100),
            1,
        )

        y += 35

        # Individual detections
        for detection in report["detections"]:

            cv2.putText(
                panel,
                f"#{detection['id']} "
                f"{detection['damage_type']}",
                (25, y),
                font,
                0.52,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            y += 28

            cv2.putText(
                panel,
                f"Confidence: "
                f"{detection['confidence'] * 100:.1f}%",
                (40, y),
                font,
                0.48,
                (190, 190, 190),
                1,
                cv2.LINE_AA,
            )

            y += 25

            cv2.putText(
                panel,
                f"Severity: "
                f"{detection['severity']}",
                (40, y),
                font,
                0.48,
                (190, 190, 190),
                1,
                cv2.LINE_AA,
            )

            y += 25

            cv2.putText(
                panel,
                f"Risk: "
                f"{detection['risk_score']:.0f}/100",
                (40, y),
                font,
                0.48,
                (190, 190, 190),
                1,
                cv2.LINE_AA,
            )

            y += 25

            cv2.putText(
                panel,
                f"Priority: "
                f"{detection['priority']}",
                (40, y),
                font,
                0.48,
                (190, 190, 190),
                1,
                cv2.LINE_AA,
            )

            y += 40

            if y > panel.shape[0] - 60:
                break

        # -----------------------------------------------
        # Combine image + panel
        # -----------------------------------------------

        combined = np.hstack(
            [image, panel]
        )

        return combined

    # --------------------------------------------------------
    # SAVE JSON REPORT
    # --------------------------------------------------------

    @staticmethod
    def save_json(
        report: Dict,
        output_path: str | Path,
    ):

        output_path = Path(output_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(
            output_path,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                report,
                f,
                indent=4,
            )

        print(
            f"JSON report saved: "
            f"{output_path}"
        )

    # --------------------------------------------------------
    # PROCESS IMAGE
    # --------------------------------------------------------

    def process_image(
        self,
        image_path: str | Path,
        output_dir: str | Path = "outputs",
        save_json: bool = True,
    ):

        image_path = Path(image_path)
        output_dir = Path(output_dir)

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        print("\n" + "=" * 60)
        print("PROCESSING IMAGE")
        print("=" * 60)
        print(f"Input: {image_path}")

        output = self.predict_image(
            image_path
        )

        annotated = self.draw_results(
            output
        )

        output_image = (
            output_dir /
            f"{image_path.stem}_roadguard.jpg"
        )

        cv2.imwrite(
            str(output_image),
            annotated,
        )

        print(
            f"Annotated image: {output_image}"
        )

        if save_json:

            output_json = (
                output_dir /
                f"{image_path.stem}_report.json"
            )

            self.save_json(
                output["report"],
                output_json,
            )

        self.print_report(
            output["report"]
        )

        return output["report"]

    # --------------------------------------------------------
    # PROCESS FOLDER
    # --------------------------------------------------------

    def process_folder(
        self,
        folder_path: str | Path,
        output_dir: str | Path = "outputs",
    ):

        folder_path = Path(folder_path)

        if not folder_path.exists():
            raise FileNotFoundError(
                f"Folder not found: {folder_path}"
            )

        extensions = {
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".webp",
        }

        images = [
            p for p in folder_path.iterdir()
            if p.suffix.lower() in extensions
        ]

        images.sort()

        print(
            f"\nFound {len(images)} images."
        )

        all_reports = []

        for i, image_path in enumerate(images):

            print(
                f"\n[{i + 1}/{len(images)}] "
                f"{image_path.name}"
            )

            try:

                report = self.process_image(
                    image_path,
                    output_dir,
                    save_json=True,
                )

                all_reports.append(
                    report
                )

            except Exception as e:

                print(
                    f"ERROR processing "
                    f"{image_path.name}: {e}"
                )

        # Save combined report
        combined_path = (
            Path(output_dir) /
            "roadguard_batch_report.json"
        )

        with open(
            combined_path,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                all_reports,
                f,
                indent=4,
            )

        print(
            f"\nCombined report saved: "
            f"{combined_path}"
        )

        return all_reports

    # --------------------------------------------------------
    # VIDEO / WEBCAM
    # --------------------------------------------------------

    def process_video(
        self,
        source,
        output_path: str | Path = "outputs/roadguard_video.mp4",
        display: bool = False,
    ):
        """
        Process a video or webcam.

        source:
            0 -> webcam
            "video.mp4" -> video file
        """

        output_path = Path(output_path)
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            raise RuntimeError(
                f"Unable to open video source: {source}"
            )

        fps = cap.get(
            cv2.CAP_PROP_FPS
        )

        if fps <= 0:
            fps = 25.0

        width = int(
            cap.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
        )

        height = int(
            cap.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
        )

        fourcc = cv2.VideoWriter_fourcc(
            *"mp4v"
        )

        writer = cv2.VideoWriter(
            str(output_path),
            fourcc,
            fps,
            (
                width + 420,
                height,
            ),
        )

        frame_number = 0

        print("\nStarting video processing...")
        print("Press Q to stop.")

        try:

            while True:

                ret, frame = cap.read()

                if not ret:
                    break

                frame_number += 1

                # YOLO accepts numpy arrays directly.
                results = self.model.predict(
                    source=frame,
                    conf=self.confidence,
                    iou=self.iou,
                    imgsz=self.image_size,
                    device=self.device,
                    verbose=False,
                )

                result = results[0]

                # Convert prediction result into
                # RoadGuard-style output.
                detections = []

                if result.boxes is not None:

                    for index in range(
                        len(result.boxes)
                    ):

                        cls_id = int(
                            result.boxes.cls[index]
                            .item()
                        )

                        confidence = float(
                            result.boxes.conf[index]
                            .item()
                        )

                        x1, y1, x2, y2 = map(
                            float,
                            result.boxes.xyxy[
                                index
                            ].cpu().numpy(),
                        )

                        damage_name = (
                            CLASS_NAMES.get(
                                cls_id,
                                f"Unknown {cls_id}",
                            )
                        )

                        area_ratio = (
                            self.calculate_area_ratio(
                                x1,
                                y1,
                                x2,
                                y2,
                                width,
                                height,
                            )
                        )

                        severity = (
                            self.calculate_severity(
                                area_ratio,
                                confidence,
                            )
                        )

                        risk_score = (
                            self.calculate_risk_score(
                                damage_name,
                                confidence,
                                area_ratio,
                            )
                        )

                        priority = (
                            self.get_priority(
                                risk_score
                            )
                        )

                        detections.append({
                            "id": index + 1,
                            "damage_type":
                                damage_name,
                            "confidence":
                                confidence,
                            "bounding_box": {
                                "x1": x1,
                                "y1": y1,
                                "x2": x2,
                                "y2": y2,
                            },
                            "severity":
                                severity,
                            "risk_score":
                                risk_score,
                            "priority":
                                priority,
                        })

                if detections:

                    overall_risk = max(
                        d["risk_score"]
                        for d in detections
                    )

                else:

                    overall_risk = 0.0

                overall_priority = (
                    self.get_priority(
                        overall_risk
                    )
                )

                frame_report = {
                    "detections": detections,
                    "detections_count":
                        len(detections),
                    "overall_risk_score":
                        overall_risk,
                    "overall_priority":
                        overall_priority,
                }

                inference_output = {
                    "report": frame_report,
                    "original_image": frame,
                }

                annotated = self.draw_results(
                    inference_output
                )

                writer.write(
                    annotated
                )

                if display:

                    cv2.imshow(
                        "RoadGuard",
                        annotated,
                    )

                    if (
                        cv2.waitKey(1)
                        & 0xFF
                    ) == ord("q"):
                        break

        finally:

            cap.release()
            writer.release()

            if display:
                cv2.destroyAllWindows()

        print(
            f"\nVideo saved to: "
            f"{output_path}"
        )

    # --------------------------------------------------------
    # PRINT REPORT
    # --------------------------------------------------------

    @staticmethod
    def print_report(
        report: Dict,
    ):

        print("\n")
        print("=" * 60)
        print("ROADGUARD ANALYSIS")
        print("=" * 60)

        print(
            f"Damage detected : "
            f"{report['detections_count']}"
        )

        print(
            f"Overall risk    : "
            f"{report['overall_risk_score']:.2f}/100"
        )

        print(
            f"Overall priority: "
            f"{report['overall_priority']}"
        )

        print("-" * 60)

        if not report["detections"]:

            print(
                "No road damage detected."
            )

        for detection in report[
            "detections"
        ]:

            print(
                f"\nDetection #{detection['id']}"
            )

            print(
                f"  Damage     : "
                f"{detection['damage_type']}"
            )

            print(
                f"  Confidence : "
                f"{detection['confidence'] * 100:.2f}%"
            )

            print(
                f"  Area       : "
                f"{detection['area_percentage']:.2f}%"
            )

            print(
                f"  Severity   : "
                f"{detection['severity']}"
            )

            print(
                f"  Risk Score : "
                f"{detection['risk_score']:.2f}/100"
            )

            print(
                f"  Priority   : "
                f"{detection['priority']}"
            )

            print(
                f"  Action     : "
                f"{detection['recommendation']}"
            )

        print("=" * 60)


# ============================================================
# COMMAND LINE INTERFACE
# ============================================================

def parse_arguments():

    parser = argparse.ArgumentParser(
        description=(
            "RoadGuard Infrastructure "
            "Intelligence - AI Road Damage Detection"
        )
    )

    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="Path to trained RoadGuard .pt model",
    )

    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help=(
            "Image, folder, video path, "
            "or webcam index such as 0"
        ),
    )

    parser.add_argument(
        "--output",
        type=str,
        default="outputs",
        help="Output directory",
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Detection confidence threshold",
    )

    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="IoU threshold for NMS",
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help=(
            "Device: cpu, cuda, 0, 1, etc. "
            "Default = automatic"
        ),
    )

    parser.add_argument(
        "--webcam",
        action="store_true",
        help="Use webcam",
    )

    parser.add_argument(
        "--display",
        action="store_true",
        help="Display video/webcam while processing",
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_arguments()

    # Create RoadGuard engine
    roadguard = RoadGuard(
        model_path=args.model,
        confidence=args.conf,
        iou=args.iou,
        image_size=args.imgsz,
        device=args.device,
    )

    source = args.source

    # --------------------------------------------------------
    # WEBCAM
    # --------------------------------------------------------

    if args.webcam:

        try:
            camera_index = int(source)
        except ValueError:
            camera_index = 0

        roadguard.process_video(
            source=camera_index,
            output_path=(
                Path(args.output) /
                "roadguard_webcam.mp4"
            ),
            display=True,
        )

        return

    source_path = Path(source)

    # --------------------------------------------------------
    # FOLDER
    # --------------------------------------------------------

    if source_path.is_dir():

        roadguard.process_folder(
            folder_path=source_path,
            output_dir=args.output,
        )

        return

    # --------------------------------------------------------
    # IMAGE
    # --------------------------------------------------------

    image_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp",
    }

    if (
        source_path.is_file()
        and source_path.suffix.lower()
        in image_extensions
    ):

        roadguard.process_image(
            image_path=source_path,
            output_dir=args.output,
            save_json=True,
        )

        return

    # --------------------------------------------------------
    # VIDEO
    # --------------------------------------------------------

    video_extensions = {
        ".mp4",
        ".avi",
        ".mov",
        ".mkv",
        ".webm",
    }

    if (
        source_path.is_file()
        and source_path.suffix.lower()
        in video_extensions
    ):

        roadguard.process_video(
            source=source_path,
            output_path=(
                Path(args.output) /
                "roadguard_video.mp4"
            ),
            display=args.display,
        )

        return

    raise ValueError(
        "\nUnsupported source.\n\n"
        "Examples:\n"
        "  image.jpg\n"
        "  images/\n"
        "  road_video.mp4\n"
        "  --webcam --source 0\n"
    )


if __name__ == "__main__":
    main()
