import os
import json
import glob
import math
import argparse
from typing import Dict, List, Optional, Tuple

import numpy as np


FINGERTIP_INDICES = {
    "thumb": 4,
    "index": 8,
    "middle": 12,
    "ring": 16,
    "pinky": 20,
}

PALM_CENTER_INDICES = [0, 5, 9, 13, 17]  # wrist + MCPs, more stable than 0:5


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_hand_json_for_pose(data_dir: str, pose_path: str) -> Optional[str]:
    """Find the matching hand landmark json for a pose json.

    Supports both of these layouts:
    1) outputs/sample_1_hand_landmarks.json
    2) outputs/sample_1_hand/sample_1_hand_landmarks.json
    3) outputs/sample_1_hand.json/sample_1_hand_landmarks.json
    4) outputs/sample_1_hand.json
    """
    pose_name = os.path.basename(pose_path)
    if not pose_name.endswith("_current_pose.json"):
        return None

    stem = pose_name[:-len("_current_pose.json")]

    candidates = [
        os.path.join(data_dir, f"{stem}_hand_landmarks.json"),
        os.path.join(data_dir, f"{stem}_hand", f"{stem}_hand_landmarks.json"),
        os.path.join(data_dir, f"{stem}_hand.json", f"{stem}_hand_landmarks.json"),
        os.path.join(data_dir, f"{stem}_hand.json"),
    ]

    for path in candidates:
        if os.path.isfile(path):
            return path

    pattern = os.path.join(data_dir, f"{stem}_hand*", "*_hand_landmarks.json")
    matches = glob.glob(pattern)
    if matches:
        return matches[0]

    return None


def infer_label_from_name(sample_id: str) -> str:
    lower = sample_id.lower()
    for label in ["palm", "claw", "fingertip"]:
        if label in lower:
            return label
    return "unknown"


def project_to_mouse_frame(point_xy: np.ndarray, pose: Dict) -> Tuple[float, float]:
    center = np.array([pose["center_x"], pose["center_y"]], dtype=np.float32)
    major = np.array([pose["major_axis_x"], pose["major_axis_y"]], dtype=np.float32)
    minor = np.array([pose["minor_axis_x"], pose["minor_axis_y"]], dtype=np.float32)
    rel = point_xy - center
    along_major = float(np.dot(rel, major))
    along_minor = float(np.dot(rel, minor))
    return along_major, along_minor


def extract_features_from_sample(hand_json_path: str, pose_json_path: str) -> Optional[Dict]:
    hand = load_json(hand_json_path)
    pose = load_json(pose_json_path)

    if not hand.get("hands"):
        return None
    if len(hand["hands"]) == 0:
        return None

    landmarks = hand["hands"][0].get("landmarks", [])
    if len(landmarks) < 21:
        return None

    features: Dict[str, float] = {}

    mouse_center = np.array([pose["center_x"], pose["center_y"]], dtype=np.float32)
    mouse_length = float(max(pose.get("mouse_length_px", 1.0), 1e-6))
    mouse_width = float(max(pose.get("mouse_width_px", 1.0), 1e-6))

    features["mouse_center_x"] = float(pose["center_x"])
    features["mouse_center_y"] = float(pose["center_y"])
    features["mouse_angle_deg"] = float(pose.get("angle_delta_deg", 0.0))
    features["mouse_length_px"] = mouse_length
    features["mouse_width_px"] = mouse_width

    wrist = landmarks[0]
    wrist_xy = np.array([wrist["x_px"], wrist["y_px"]], dtype=np.float32)
    features["wrist_x"] = float(wrist_xy[0])
    features["wrist_y"] = float(wrist_xy[1])
    features["wrist_to_mouse_dx"] = float(wrist_xy[0] - mouse_center[0])
    features["wrist_to_mouse_dy"] = float(wrist_xy[1] - mouse_center[1])
    features["wrist_to_mouse_distance"] = float(np.linalg.norm(wrist_xy - mouse_center))

    wrist_major, wrist_minor = project_to_mouse_frame(wrist_xy, pose)
    features["wrist_major"] = wrist_major
    features["wrist_minor"] = wrist_minor
    features["wrist_major_norm"] = wrist_major / mouse_length
    features["wrist_minor_norm"] = wrist_minor / mouse_width

    palm_center = np.mean(
        np.array([[landmarks[i]["x_px"], landmarks[i]["y_px"]] for i in PALM_CENTER_INDICES], dtype=np.float32),
        axis=0,
    )
    features["palm_center_x"] = float(palm_center[0])
    features["palm_center_y"] = float(palm_center[1])
    palm_major, palm_minor = project_to_mouse_frame(palm_center, pose)
    features["palm_major"] = palm_major
    features["palm_minor"] = palm_minor
    features["palm_major_norm"] = palm_major / mouse_length
    features["palm_minor_norm"] = palm_minor / mouse_width

    for name, idx in FINGERTIP_INDICES.items():
        pt = np.array([landmarks[idx]["x_px"], landmarks[idx]["y_px"]], dtype=np.float32)
        features[f"fingertip_{name}_x"] = float(pt[0])
        features[f"fingertip_{name}_y"] = float(pt[1])
        features[f"fingertip_{name}_to_mouse_dist"] = float(np.linalg.norm(pt - mouse_center))

        along_major, along_minor = project_to_mouse_frame(pt, pose)
        features[f"fingertip_{name}_major"] = along_major
        features[f"fingertip_{name}_minor"] = along_minor
        features[f"fingertip_{name}_major_norm"] = along_major / mouse_length
        features[f"fingertip_{name}_minor_norm"] = along_minor / mouse_width

        reach = float(np.linalg.norm(pt - palm_center))
        features[f"finger_{name}_reach"] = reach
        features[f"finger_{name}_reach_norm"] = reach / mouse_length

    # Global quality/debug fields that are useful later.
    for key in [
        "middle_marker_error_px",
        "angle_delta_deg",
        "marker_scale",
    ]:
        if key in pose:
            features[key] = float(pose[key]) if pose[key] is not None else math.nan

    if "pose_stable" in pose:
        features["pose_stable"] = 1.0 if pose["pose_stable"] else 0.0

    return features


def batch_extract_features(data_dir: str, output_file: str) -> List[Dict]:
    pose_files = sorted(glob.glob(os.path.join(data_dir, "*_current_pose.json")))
    all_samples: List[Dict] = []

    for pose_file in pose_files:
        sample_id = os.path.basename(pose_file)[:-len("_current_pose.json")]
        hand_file = find_hand_json_for_pose(data_dir, pose_file)
        if hand_file is None:
            print(f"跳过 {sample_id}: 找不到匹配的 hand landmarks json")
            continue

        try:
            features = extract_features_from_sample(hand_file, pose_file)
            if features is None:
                print(f"跳过 {sample_id}: hand json 中没有有效手部关键点")
                continue

            record = {
                "sample_id": sample_id,
                "label": infer_label_from_name(sample_id),
                "pose_json": os.path.relpath(pose_file),
                "hand_json": os.path.relpath(hand_file),
                "features": features,
            }
            all_samples.append(record)
            print(f"✓ 提取成功: {sample_id}")
        except Exception as e:
            print(f"✗ 提取失败 {sample_id}: {e}")

    output_path = output_file
    if not os.path.isabs(output_path):
        output_path = os.path.join(os.getcwd(), output_file)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_samples, f, indent=2, ensure_ascii=False)

    print(f"\n完成！共提取 {len(all_samples)} 个样本")
    print(f"保存到: {output_path}")
    return all_samples


def main():
    parser = argparse.ArgumentParser(description="Extract grip features from outputs directory.")
    parser.add_argument("--data_dir", default="outputs", help="Directory containing *_current_pose.json and hand landmark jsons")
    parser.add_argument("--output", default="features.json", help="Output json path")
    args = parser.parse_args()

    batch_extract_features(args.data_dir, args.output)


if __name__ == "__main__":
    main()
