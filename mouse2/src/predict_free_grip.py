import argparse
import json
import os
import re
from pathlib import Path
from collections import Counter, defaultdict

import joblib
import numpy as np


LABELS = ["claw", "fingertip", "palm"]


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def safe_float(v, default=0.0):
    try:
        if v is None:
            return float(default)
        return float(v)
    except Exception:
        return float(default)


def get(features, name, default=0.0):
    return safe_float(features.get(name, default), default)


def flatten_features(features):
    return {
        k: float(v)
        for k, v in features.items()
        if isinstance(v, (int, float)) and v is not None
    }


def add_relative_structure_features(f):
    """
    Must match the extra feature engineering used in build_training_dataset_3mouse_v2_relative_speedfilter.py.
    """
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


def infer_meta_from_path(path):
    parts = list(Path(path).parts)
    valid_mice = {"G102", "X2H", "XliteV3ES", "XliteCrazyLight"}

    user_id = "unknown_user"
    mouse_id = "unknown_mouse"
    condition = "unknown"

    # 1) Parse from folder structure if available:
    # test/outputs/user_001/G102/free/Condition B/...
    for part in parts:
        low = part.lower()

        if low.startswith("user_"):
            # Avoid taking the full filename as user id.
            m = re.match(r"^(user_\d+)$", part, flags=re.I)
            if m:
                user_id = m.group(1)

        if part in valid_mice:
            mouse_id = part

        if low in ["condition a", "condition_a", "conditiona"]:
            condition = "A"
        elif low in ["condition b", "condition_b", "conditionb"]:
            condition = "B"
        elif low in ["condition c", "condition_c", "conditionc"]:
            condition = "C"
        elif low in ["condition d", "condition_d", "conditiond"]:
            condition = "D"

    # 2) Parse from flat filename:
    # user_001_G102_free_B_session_01_trial_features_v2.json
    name = Path(path).name

    m = re.search(r"(user_\d+)", name, flags=re.I)
    if m:
        user_id = m.group(1)

    for mouse in valid_mice:
        if mouse.lower() in name.lower():
            mouse_id = mouse
            break

    for c in ["A", "B", "C", "D"]:
        if f"_{c}_" in name:
            condition = c
            break

    return user_id, mouse_id, condition


def load_feature_columns(model_path, feature_columns_path=None):
    if feature_columns_path:
        return load_json(feature_columns_path)

    candidates = [
        Path(model_path).parent / "feature_columns_3mouse.json",
        Path(model_path).parent / "feature_columns.json",
        Path(model_path).parent / "feature_columns_2stage.json",
        Path("models/trained/feature_columns_3mouse.json"),
        Path("models/trained/feature_columns.json"),
    ]

    for p in candidates:
        if p.exists():
            return load_json(p)

    raise FileNotFoundError(
        "feature columns JSON not found. Please pass --feature_columns "
        "such as models/trained/feature_columns_3mouse.json"
    )


def load_label_names(model_path, label_encoder_path=None):
    """
    Load class names for models trained with encoded labels.
    Expected file: models/trained/label_encoder_3mouse.json
    """
    if label_encoder_path:
        data = load_json(label_encoder_path)
        if isinstance(data, dict) and "classes" in data:
            return [str(x) if str(x) != "finger" else "fingertip" for x in data["classes"]]
        if isinstance(data, list):
            return [str(x) if str(x) != "finger" else "fingertip" for x in data]

    candidates = [
        Path(model_path).parent / "label_encoder_3mouse.json",
        Path(model_path).parent / "label_encoder.json",
        Path("models/trained/label_encoder_3mouse.json"),
        Path("models/trained/label_encoder.json"),
    ]

    for p in candidates:
        if p.exists():
            data = load_json(p)
            if isinstance(data, dict) and "classes" in data:
                return [str(x) if str(x) != "finger" else "fingertip" for x in data["classes"]]
            if isinstance(data, list):
                return [str(x) if str(x) != "finger" else "fingertip" for x in data]

    return LABELS


def model_label_order(model, label_names=None):
    if label_names is None:
        label_names = LABELS

    if hasattr(model, "classes_"):
        out = []
        for c in model.classes_:
            # If model classes are numeric 0/1/2, map them through label_names.
            try:
                idx = int(c)
                if 0 <= idx < len(label_names):
                    out.append(label_names[idx])
                    continue
            except Exception:
                pass

            name = str(c)
            if name == "finger":
                name = "fingertip"
            out.append(name)
        return out

    return label_names


def predict_trial(model, feature_columns, features, label_names=None):
    x = np.array([[float(features.get(k, 0.0)) for k in feature_columns]], dtype=np.float32)

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x)[0]
        classes = model_label_order(model, label_names)

        prob = {k: 0.0 for k in LABELS}
        for cls, p in zip(classes, proba):
            prob[cls] = float(p)

        pred = max(prob.items(), key=lambda kv: kv[1])[0]
        return pred, prob

    pred = str(model.predict(x)[0])
    if pred == "finger":
        pred = "fingertip"

    prob = {k: 0.0 for k in LABELS}
    prob[pred] = 1.0
    return pred, prob


def normalize_prob(prob_sum):
    total = sum(prob_sum.values())
    if total <= 1e-9:
        return {k: 0.0 for k in LABELS}
    return {k: float(v / total) for k, v in prob_sum.items()}

def decide_tendency(prob, hybrid_gap=0.20, min_second=0.25):
    ordered = sorted(prob.items(), key=lambda kv: kv[1], reverse=True)

    top, top_v = ordered[0]
    second, second_v = ordered[1]

    if second_v >= min_second and (top_v - second_v) <= hybrid_gap:
        combo = {top, second}

        # Palm + Fingertip is not treated as a valid hybrid tendency.
        # Use the higher-probability class instead.
        if combo == {"palm", "fingertip"}:
            return {
                "type": "dominant",
                "main": top,
                "secondary": None,
                "description": top,
            }

        return {
            "type": "hybrid",
            "main": top,
            "secondary": second,
            "description": f"{top} + {second}",
        }

    return {
        "type": "dominant",
        "main": top,
        "secondary": None,
        "description": top,
    }


def recommendation_placeholder(tendency):
    """
    Prototype recommendation rule.

    G102 is used only as the baseline mouse for detecting natural grip tendency.
    It is NOT used as a recommendation target.

    Current recommendation candidates:
    - Palm      -> XliteCrazyLight
    - Claw      -> X2H
    - Fingertip -> XliteV3ES

    For hybrid tendencies, the recommendation follows the stronger/main grip.
    """
    main = tendency["main"]
    sec = tendency.get("secondary")

    grip_to_mouse = {
        "palm": "XliteCrazyLight",
        "claw": "X2H",
        "fingertip": "XliteV3ES",
    }

    recommended_mouse = grip_to_mouse.get(main, "undetermined")

    if sec is None:
        reason = f"{main} dominant tendency."
    else:
        reason = (
            f"{main} + {sec} hybrid tendency. "
            f"Since {main} is the stronger tendency, recommend the mouse for {main}."
        )

    return {
        "note": "Prototype only; validate using task performance and questionnaire results.",
        "recommended_mouse": recommended_mouse,
        "reason": reason,
    }


def predict_file(path, model, feature_columns, label_names=None, only_stable=True, max_peak_speed=None,
                 hybrid_gap=0.15, min_second=0.25):
    data = load_json(path)
    user_id, mouse_id, condition = infer_meta_from_path(path)

    pred_counts = Counter()
    prob_sum = {k: 0.0 for k in LABELS}
    trial_results = []
    skipped = 0
    skipped_by_speed = 0

    for trial in data.get("trials", []):
        features = flatten_features(trial.get("features", {}))

        if not features:
            skipped += 1
            continue

        if only_stable and features.get("pose_stable", 1.0) < 1.0:
            skipped += 1
            continue

        if max_peak_speed is not None and get(features, "mouse_peak_speed", 0.0) > max_peak_speed:
            skipped += 1
            skipped_by_speed += 1
            continue

        features = add_relative_structure_features(features)

        pred, prob = predict_trial(model, feature_columns, features, label_names=label_names)
        pred_counts[pred] += 1

        for k, v in prob.items():
            prob_sum[k] += v

        trial_results.append({
            "trial_index": trial.get("trial_index"),
            "target_id": trial.get("target_id"),
            "prediction": pred,
            "probability": prob,
            "mouse_peak_speed": get(features, "mouse_peak_speed", 0.0),
            "marker_ok_ratio": get(features, "marker_ok_ratio", 0.0),
            "hand_ok_ratio": get(features, "hand_ok_ratio", 0.0),
        })

    n = len(trial_results)
    mean_prob = normalize_prob(prob_sum)
    tendency = decide_tendency(mean_prob, hybrid_gap=hybrid_gap, min_second=min_second)

    return {
        "source_file": str(path),
        "user_id": user_id,
        "mouse_id": mouse_id,
        "condition": condition,
        "n_trials_used": n,
        "n_trials_skipped": skipped,
        "n_trials_skipped_by_speed": skipped_by_speed,
        "prediction_counts": dict(pred_counts),
        "prediction_ratio_by_vote": {
            k: float(pred_counts.get(k, 0) / n) if n > 0 else 0.0
            for k in LABELS
        },
        "mean_probability": mean_prob,
        "tendency": tendency,
        "recommendation_placeholder": recommendation_placeholder(tendency),
        "trial_results": trial_results,
    }


def aggregate_by_user(file_results, hybrid_gap=0.15, min_second=0.25):
    grouped = defaultdict(list)

    for item in file_results:
        grouped[item["user_id"]].append(item)

    summary = {}

    for user_id, items in grouped.items():
        total_trials = sum(x["n_trials_used"] for x in items)
        prob_sum = {k: 0.0 for k in LABELS}
        vote_counts = Counter()

        for item in items:
            n = item["n_trials_used"]
            for k in LABELS:
                prob_sum[k] += item["mean_probability"].get(k, 0.0) * n
                vote_counts[k] += item["prediction_counts"].get(k, 0)

        if total_trials > 0:
            mean_prob = {k: float(v / total_trials) for k, v in prob_sum.items()}
        else:
            mean_prob = {k: 0.0 for k in LABELS}

        tendency = decide_tendency(mean_prob, hybrid_gap=hybrid_gap, min_second=min_second)

        summary[user_id] = {
            "n_files": len(items),
            "n_trials_used": total_trials,
            "prediction_counts": dict(vote_counts),
            "prediction_ratio_by_vote": {
                k: float(vote_counts.get(k, 0) / total_trials) if total_trials > 0 else 0.0
                for k in LABELS
            },
            "mean_probability": mean_prob,
            "tendency": tendency,
            "recommendation_placeholder": recommendation_placeholder(tendency),
        }

    return summary


def main():
    parser = argparse.ArgumentParser(description="Predict free/natural grip tendency using trained grip classifier.")
    parser.add_argument("--input_root", required=True, help="Folder containing free *_trial_features_v2.json files.")
    parser.add_argument("--baseline_mouse", default="G102", help="Only use this mouse to infer natural grip tendency. Default: G102.")
    parser.add_argument("--model", default="models/trained/grip_model_3mouse.pkl")
    parser.add_argument("--feature_columns", default=None)
    parser.add_argument("--label_encoder", default=None, help="Optional label encoder JSON, e.g. models/trained/label_encoder_3mouse.json")
    parser.add_argument("--output", default="data/datasets/free_prediction_report.json")
    parser.add_argument("--include_unstable", action="store_true")
    parser.add_argument("--max_peak_speed", type=float, default=None)
    parser.add_argument("--hybrid_gap", type=float, default=0.20)
    parser.add_argument("--min_second", type=float, default=0.25)
    args = parser.parse_args()

    model = joblib.load(args.model)
    if isinstance(model, dict) and "model" in model:
        model = model["model"]

    feature_columns = load_feature_columns(args.model, args.feature_columns)
    label_names = load_label_names(args.model, args.label_encoder)

    files = sorted(Path(args.input_root).rglob("*_trial_features_v2.json"))

    # Use only the baseline mouse videos to infer user's natural grip tendency.
    # Default: G102.
    if args.baseline_mouse:
        baseline = args.baseline_mouse.lower()
        files = [
            f for f in files
            if baseline in f.name.lower() or any(baseline == part.lower() for part in f.parts)
        ]

    if not files:
        print("[ERROR] No *_trial_features_v2.json found under:", args.input_root)
        print("[ERROR] baseline_mouse:", args.baseline_mouse)
        return

    file_results = []

    for file in files:
        r = predict_file(
            file,
            model=model,
            feature_columns=feature_columns,
            label_names=label_names,
            only_stable=(not args.include_unstable),
            max_peak_speed=args.max_peak_speed,
            hybrid_gap=args.hybrid_gap,
            min_second=args.min_second,
        )
        file_results.append(r)

        p = r["mean_probability"]
        rec_mouse = r["recommendation_placeholder"].get("recommended_mouse", "unknown")
        print(
            f"{r['user_id']} | {r['mouse_id']} | cond={r['condition']} | "
            f"n={r['n_trials_used']} | "
            f"claw={p['claw']:.2f}, fingertip={p['fingertip']:.2f}, palm={p['palm']:.2f} | "
            f"tendency={r['tendency']['description']} | recommended={rec_mouse}"
        )

    user_summary = aggregate_by_user(
        file_results,
        hybrid_gap=args.hybrid_gap,
        min_second=args.min_second,
    )

    report = {
        "meta": {
            "input_root": args.input_root,
            "baseline_mouse": args.baseline_mouse,
            "model": args.model,
            "feature_columns_count": len(feature_columns),
            "label_names": label_names,
            "n_files": len(files),
            "include_unstable": args.include_unstable,
            "max_peak_speed": args.max_peak_speed,
            "hybrid_gap": args.hybrid_gap,
            "min_second": args.min_second,
            "note": "Free grip is evaluated as tendency distribution, not single-label accuracy."
        },
        "file_results": file_results,
        "user_summary": user_summary,
    }

    save_json(args.output, report)

    print("\n=== User Summary ===")
    for user_id, s in sorted(user_summary.items()):
        p = s["mean_probability"]
        rec_mouse = s["recommendation_placeholder"].get("recommended_mouse", "unknown")
        print(
            f"{user_id}: claw={p['claw']:.2f}, fingertip={p['fingertip']:.2f}, palm={p['palm']:.2f}, "
            f"tendency={s['tendency']['description']}, recommended={rec_mouse}"
        )

    print("\nSaved:", args.output)


if __name__ == "__main__":
    main()
