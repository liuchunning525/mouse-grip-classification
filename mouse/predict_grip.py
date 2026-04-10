import argparse
import subprocess
import os
import json
import joblib
import numpy as np


# ======================
# 推荐系统
# ======================
def recommend_mouse(grip):
    if grip == "palm":
        return [
            "Zowie EC2 / EC1（高）",
            "Logitech G703（贴）",
            "Razer DeathAdder"
        ]
    elif grip == "claw":
        return [
            "Logitech G Pro X Superlight",
            "Razer Viper V2 Pro",
            "Zowie ZA13"
        ]
    elif grip == "fingertip":
        return [
            "Finalmouse Starlight",
            "Razer Viper Mini",
            "G-Wolves HSK"
        ]
    return ["Unknown"]


# ======================
# 从 features 提取向量
# ======================
def feature_dict_to_vector(features):
    return np.array(list(features.values())).reshape(1, -1)


# ======================
# 主流程
# ======================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    args = parser.parse_args()

    image_path = args.image
    name = os.path.splitext(os.path.basename(image_path))[0]

    os.makedirs("temp", exist_ok=True)

    print("Step 1: Extract hand landmarks...")
    subprocess.run([
        "python", "extract_hand_landmarks.py",
        "--image", image_path,
        "--output", f"temp/{name}_hand"
    ], check=True)

    hand_json = f"temp/{name}_hand/{name}_hand_landmarks.json"

    print("Step 2: Refine mouse pose...")
    subprocess.run([
        "python", "refine_mouse_pose_with_hand.py",
        "--ref_image", "images/mouse_ref.jpg",
        "--cur_image", image_path,
        "--ref_pose", "outputs/mouse_pose_ref.json",
        "--work_area_json", "outputs/mouse_ref_work_area.json",
        "--hand_json", hand_json,
        "--output_dir", "temp",
        "--output_name", f"{name}_current_pose.json"
    ], check=True)

    pose_path = f"temp/{name}_current_pose.json"

    print("Step 3: Extract features...")

    # ===== 手动提特征（轻量版）=====
    with open(pose_path, "r", encoding="utf-8") as f:
        pose_data = json.load(f)

    hand = pose_data["hands"][0]

    def get_lm(id):
        for lm in hand["landmarks_mouse_frame"]:
            if lm["id"] == id:
                return lm
        return None

    WRIST = 0
    INDEX_TIP = 8
    MIDDLE_TIP = 12
    RING_TIP = 16
    PINKY_TIP = 20

    wrist = get_lm(WRIST)
    tips = [
        get_lm(INDEX_TIP),
        get_lm(MIDDLE_TIP),
        get_lm(RING_TIP),
        get_lm(PINKY_TIP)
    ]

    palm_position = wrist["a_norm"]

    finger_forward = sum([t["a_norm"] for t in tips]) / 4.0

    def dist(lm):
        dx = lm["a_norm"] - wrist["a_norm"]
        dy = lm["b_norm"] - wrist["b_norm"]
        return (dx**2 + dy**2) ** 0.5

    finger_spread = sum([dist(t) for t in tips]) / 4.0

    features = {
        "palm_position": palm_position,
        "finger_forward": finger_forward,
        "finger_spread": finger_spread
    }

    X = feature_dict_to_vector(features)

    print("Step 4: Load model...")
    model = joblib.load("grip_model.pkl")

    print("Step 5: Predict...")
    pred = model.predict(X)[0]

    # 如果模型是编码的
    try:
        le = joblib.load("label_encoder.pkl")
        pred = le.inverse_transform([pred])[0]
    except:
        pass

    print("\n=== Result ===")
    print(f"Grip: {pred}")

    try:
        prob = model.predict_proba(X)[0]
        confidence = max(prob)
        print(f"Confidence: {confidence:.2f}")
    except:
        print("Confidence: N/A")

    print("\nRecommended mouse:")
    for m in recommend_mouse(pred):
        print("-", m)


if __name__ == "__main__":
    main()