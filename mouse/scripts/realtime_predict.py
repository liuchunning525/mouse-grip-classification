import os
import cv2
import time
import joblib
import numpy as np
import mediapipe as mp
import pandas as pd
from collections import Counter, deque

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ========= Config =========
NUM_HANDS = 1
VIDEO_SOURCE = 0   # 0 = webcam, or use "test.mp4"
SMOOTH_WINDOW = 10


def get_project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def create_detector(model_path: str, num_hands: int = 1):
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=num_hands
    )
    return vision.HandLandmarker.create_from_options(options)


def get_recommendation(label: str) -> str:
    recommendations = {
        "palm": "Larger mouse, stronger palm support",
        "claw": "Medium size, higher back hump",
        "fingertip": "Smaller and lighter mouse"
    }
    return recommendations.get(label, "No recommendation available")


def extract_features_from_result(detection_result):
    """
    Extract 63-dim flattened landmark features from the first detected hand.
    Return a pandas DataFrame with proper feature names, or None if no hand.
    """
    if not detection_result.hand_landmarks:
        return None

    hand_landmarks = detection_result.hand_landmarks[0]
    features = []

    for lm in hand_landmarks:
        features.extend([lm.x, lm.y, lm.z])

    columns = []
    for i in range(21):
        columns.extend([f"x{i}", f"y{i}", f"z{i}"])

    return pd.DataFrame([features], columns=columns)


def draw_landmarks_on_frame(frame_bgr, detection_result):

    annotated = frame_bgr.copy()

    if not detection_result.hand_landmarks:
        return annotated

    height, width, _ = annotated.shape

    # 手部21点连接关系
    HAND_CONNECTIONS = [
        (0,1),(1,2),(2,3),(3,4),
        (0,5),(5,6),(6,7),(7,8),
        (5,9),(9,10),(10,11),(11,12),
        (9,13),(13,14),(14,15),(15,16),
        (13,17),(17,18),(18,19),(19,20),
        (0,17)
    ]

    for hand_landmarks in detection_result.hand_landmarks:

        points = []

        for lm in hand_landmarks:
            x = int(lm.x * width)
            y = int(lm.y * height)
            points.append((x, y))

        # 画点
        for point in points:
            cv2.circle(annotated, point, 4, (0,255,0), -1)

        # 画骨架线
        for start, end in HAND_CONNECTIONS:
            cv2.line(annotated, points[start], points[end], (255,0,0), 2)

    return annotated

def get_smoothed_label(history):
    if not history:
        return None
    return Counter(history).most_common(1)[0][0]


def main():
    project_root = get_project_root()
    detector_model_path = os.path.join(project_root, "hand_landmarker.task")
    classifier_path = os.path.join(project_root, "models", "grip_model.pkl")

    if not os.path.exists(detector_model_path):
        print(f"[ERROR] Hand landmarker model not found: {detector_model_path}")
        return

    if not os.path.exists(classifier_path):
        print(f"[ERROR] Trained classifier not found: {classifier_path}")
        print("Run train_classifier.py first.")
        return

    detector = create_detector(detector_model_path, NUM_HANDS)
    clf = joblib.load(classifier_path)

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video source: {VIDEO_SOURCE}")
        return

    prediction_history = deque(maxlen=SMOOTH_WINDOW)

    prev_time = time.time()
    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] Video ended or failed to read frame.")
            break

        # 如果是摄像头，翻转更自然；如果是视频文件，也翻转会不太合适
        if VIDEO_SOURCE == 0:
            frame = cv2.flip(frame, 1)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        detection_result = detector.detect(mp_image)
        annotated_frame = draw_landmarks_on_frame(frame, detection_result)

        features = extract_features_from_result(detection_result)

        display_label = None
        confidence_text = ""

        if features is not None:
            pred_label = clf.predict(features)[0]
            prediction_history.append(pred_label)
            display_label = get_smoothed_label(prediction_history)

            if hasattr(clf, "predict_proba"):
                probs = clf.predict_proba(features)[0]
                confidence = float(np.max(probs))
                confidence_text = f"{confidence:.2f}"

        # FPS
        current_time = time.time()
        fps = 1.0 / (current_time - prev_time + 1e-6)
        prev_time = current_time

        if display_label is not None:
            recommendation = get_recommendation(display_label)

            cv2.putText(
                annotated_frame,
                f"Grip: {display_label}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2
            )

            if confidence_text:
                cv2.putText(
                    annotated_frame,
                    f"Confidence: {confidence_text}",
                    (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )

            cv2.putText(
                annotated_frame,
                f"Recommend: {recommendation}",
                (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
        else:
            cv2.putText(
                annotated_frame,
                "No hand detected",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2
            )

        cv2.putText(
            annotated_frame,
            f"FPS: {fps:.1f}",
            (20, 145),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2
        )

        cv2.imshow("Mouse Grip Video Prediction", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()