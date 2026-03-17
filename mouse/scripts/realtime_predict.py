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
SHOW_WORLD_COORDS = True
TARGET_LANDMARK_INDEX = 8 

def get_project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_homography(project_root: str):
    h_path = os.path.join(project_root, "camera", "H_matrix.npy")
    if not os.path.exists(h_path):
        print(f"[ERROR] Homography matrix not found: {h_path}")
        return None
    return np.load(h_path)


def pixel_to_world(x, y, H):
    p = np.array([x, y, 1.0], dtype=np.float32)
    w = H @ p
    w = w / w[2]
    return float(w[0]), float(w[1])


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

    H = None
    if SHOW_WORLD_COORDS:
        H = load_homography(project_root)
        if H is None:
            return

    detector = create_detector(detector_model_path, NUM_HANDS)
    clf = joblib.load(classifier_path)

    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video source: {VIDEO_SOURCE}")
        return

    prediction_history = deque(maxlen=SMOOTH_WINDOW)

    prev_time = time.time()
    print("Press 'q' to quit.")

    landmark_names = {
        0: "wrist",
        1: "thumb_cmc",
        2: "thumb_mcp",
        3: "thumb_ip",
        4: "thumb_tip",
        5: "index_mcp",
        6: "index_pip",
        7: "index_dip",
        8: "index_tip",
        9: "middle_mcp",
        10: "middle_pip",
        11: "middle_dip",
        12: "middle_tip",
        13: "ring_mcp",
        14: "ring_pip",
        15: "ring_dip",
        16: "ring_tip",
        17: "pinky_mcp",
        18: "pinky_pip",
        19: "pinky_dip",
        20: "pinky_tip",
    }

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

        important_landmarks_text = []
        landmark_world_coords = {}

        if detection_result.hand_landmarks:
            height, width, _ = frame.shape
            hand_landmarks = detection_result.hand_landmarks[0]

            # 以 wrist 的 z 作为参考
            wrist_z = hand_landmarks[0].z if USE_WRIST_AS_Z_REF else 0.0

            for i, lm in enumerate(hand_landmarks):
                api_x = lm.x
                api_y = lm.y
                api_z = lm.z

                pixel_x = api_x * width
                pixel_y = api_y * height

                real_x, real_y = pixel_to_world(pixel_x, pixel_y, H)

                # MediaPipe z: closer to camera is usually more negative
                # 为了让“更高/更靠近摄像头”显示为正值，这里取负号
                scaled_z = -(api_z - wrist_z) * Z_SCALE

                landmark_world_coords[i] = {
                    "name": landmark_names.get(i, f"lm_{i}"),
                    "api_x": api_x,
                    "api_y": api_y,
                    "api_z": api_z,
                    "pixel_x": pixel_x,
                    "pixel_y": pixel_y,
                    "real_x": real_x,
                    "real_y": real_y,
                    "scaled_z": scaled_z,
                }

            for idx in range(21):
                if idx not in landmark_world_coords:
                    continue

                info = landmark_world_coords[idx]
                px = int(info["pixel_x"])
                py = int(info["pixel_y"])
                real_x = info["real_x"]
                real_y = info["real_y"]
                scaled_z = info["scaled_z"]

                cv2.putText(
                    annotated_frame,
                    f"{idx}:({real_x:.1f},{real_y:.1f},{scaled_z:.1f})",
                    (px + 6, py - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (0, 255, 255),
                    1
                )


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
        

        #base_y = 180
        #for idx, text in enumerate(important_landmarks_text):
        #    cv2.putText(
        #        annotated_frame,
        #        text,
        #        (20, base_y + idx * 18),
        #        cv2.FONT_HERSHEY_SIMPLEX,
        #       0.4,
        #       (255, 255, 255),
        #         1
        #    )

        cv2.imshow("Mouse Grip Video Prediction", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()