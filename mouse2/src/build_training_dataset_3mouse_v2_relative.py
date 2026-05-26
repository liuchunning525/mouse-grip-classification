import argparse
import json
import os
from pathlib import Path

VALID_LABELS = {"palm", "claw", "fingertip", "finger"}
LABEL_ALIAS = {
    "finger": "fingertip"
}
VALID_MICE = {"G102", "XliteV3ES", "X2H"}

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def infer_label_from_path(path):
    parts = [p.lower() for p in Path(path).parts]
    for label in VALID_LABELS:
        if label in parts:
            return label
    name = Path(path).name.lower()
    for label in VALID_LABELS:
        if f"_{label}_" in name:
            return label
    return None

def infer_mouse_from_path_or_data(path, data=None):
    if data:
        mouse_id = data.get("mouse_id")
        if mouse_id:
            return str(mouse_id)

    for part in Path(path).parts:
        if part in VALID_MICE:
            return part

    name = Path(path).name
    for mouse in VALID_MICE:
        if mouse.lower() in name.lower():
            return mouse

    return "unknown_mouse"

def infer_user_from_path(path):
    for part in Path(path).parts:
        if part.lower().startswith("user_"):
            return part
    return "unknown_user"

def infer_condition_from_path_or_data(path, data=None):
    if data:
        cond = data.get("condition")
        if cond:
            return str(cond)

    for part in Path(path).parts:
        low = part.lower()
        if low in ["condition a", "condition_a"]:
            return "A"
        if low in ["condition b", "condition_b"]:
            return "B"

    name = Path(path).name
    if "_A_" in name:
        return "A"
    if "_B_" in name:
        return "B"

    return "unknown"

def flatten_features(features):
    out = {}
    for k, v in features.items():
        if isinstance(v, (int, float)) and v is not None:
            out[k] = float(v)
    return out

def get(features, name, default=0.0):
    try:
        return float(features.get(name, default))
    except Exception:
        return float(default)

def add_relative_structure_features(f):
    fingers = ["thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip"]

    for finger in fingers:
        for axis in ["major", "minor"]:
            for stat in ["mean", "min", "max"]:
                finger_key = f"{finger}_rel_{axis}_{stat}"
                palm_key = f"palm_rel_{axis}_{stat}"
                wrist_key = f"wrist_rel_{axis}_{stat}"

                f[f"{finger}_minus_palm_{axis}_{stat}"] = get(f, finger_key) - get(f, palm_key)
                f[f"{finger}_minus_wrist_{axis}_{stat}"] = get(f, finger_key) - get(f, wrist_key)

            finger_std = get(f, f"{finger}_rel_{axis}_std")
            palm_std = get(f, f"palm_rel_{axis}_std")
            wrist_std = get(f, f"wrist_rel_{axis}_std")

            f[f"{finger}_to_palm_std_ratio_{axis}"] = finger_std / (palm_std + 1e-6)
            f[f"{finger}_to_wrist_std_ratio_{axis}"] = finger_std / (wrist_std + 1e-6)

    finger_pairs = [
        ("index_tip", "middle_tip"),
        ("index_tip", "ring_tip"),
        ("index_tip", "pinky_tip"),
        ("middle_tip", "ring_tip"),
        ("middle_tip", "pinky_tip"),
    ]

    for a, b in finger_pairs:
        for axis in ["major", "minor"]:
            for stat in ["mean", "min", "max", "std"]:
                f[f"{a}_minus_{b}_{axis}_{stat}"] = (
                    get(f, f"{a}_rel_{axis}_{stat}") -
                    get(f, f"{b}_rel_{axis}_{stat}")
                )

    reach_fingers = ["thumb", "index", "middle", "ring", "pinky"]

    for a in reach_fingers:
        for b in reach_fingers:
            if a == b:
                continue
            for stat in ["mean", "min", "max", "std"]:
                ka = f"finger_{a}_reach_norm_{stat}"
                kb = f"finger_{b}_reach_norm_{stat}"
                f[f"finger_{a}_minus_{b}_reach_norm_{stat}"] = get(f, ka) - get(f, kb)
                f[f"finger_{a}_to_{b}_reach_norm_ratio_{stat}"] = get(f, ka) / (get(f, kb) + 1e-6)

    for finger in ["index_tip", "middle_tip", "ring_tip", "pinky_tip"]:
        f[f"{finger}_motion_to_palm_path_ratio"] = (
            get(f, f"{finger}_relative_motion") / (get(f, "palm_path_length") + 1e-6)
        )
        f[f"{finger}_motion_to_wrist_path_ratio"] = (
            get(f, f"{finger}_relative_motion") / (get(f, "wrist_path_length") + 1e-6)
        )

    return f

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs_root", default="data/outputs")
    parser.add_argument("--output", default="data/datasets/training_dataset_3mouse_v2_relative.json")
    parser.add_argument("--only_stable", action="store_true")
    args = parser.parse_args()

    samples = []
    skipped = 0
    files = list(Path(args.outputs_root).rglob("*_trial_features_v2.json"))

    for file in files:
        data = load_json(file)

        label = infer_label_from_path(file)
        if label is None:
            label = data.get("grip_label")

        label = LABEL_ALIAS.get(label, label)

        if label not in VALID_LABELS:
            print("[SKIP] Cannot infer grip label:", file)
            skipped += 1
            continue

        user_id = infer_user_from_path(file)
        mouse_id = infer_mouse_from_path_or_data(file, data)
        condition = infer_condition_from_path_or_data(file, data)

        for trial in data.get("trials", []):
            features = flatten_features(trial.get("features", {}))
            if not features:
                skipped += 1
                continue

            if args.only_stable and features.get("pose_stable", 1.0) < 1.0:
                skipped += 1
                continue

            features = add_relative_structure_features(features)

            samples.append({
                "source_file": str(file),
                "user_id": user_id,
                "mouse_id": mouse_id,
                "label": label,
                "condition": condition,
                "trial_index": trial.get("trial_index"),
                "target_id": trial.get("target_id"),
                "features": features
            })

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    print("=== Build 3-Mouse Training Dataset Done ===")
    print("Input files:", len(files))
    print("Samples:", len(samples))
    print("Skipped:", skipped)

    label_counts = {}
    mouse_counts = {}
    for s in samples:
        label_counts[s["label"]] = label_counts.get(s["label"], 0) + 1
        mouse_counts[s["mouse_id"]] = mouse_counts.get(s["mouse_id"], 0) + 1

    print("Label counts:", label_counts)
    print("Mouse counts:", mouse_counts)
    print("Saved:", args.output)

if __name__ == "__main__":
    main()
