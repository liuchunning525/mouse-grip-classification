import argparse
import json
import os
import re
import math
from pathlib import Path
from collections import defaultdict


VALID_MICE = ["G102", "X2H", "XliteV3ES"]


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


def infer_meta_from_path(path):
    path = Path(path)
    name = path.name

    user_id = "unknown_user"
    mouse_id = "unknown_mouse"
    condition = "unknown"

    m = re.search(r"(user_\d+)", name, flags=re.I)
    if m:
        user_id = m.group(1)

    for mouse in VALID_MICE:
        if mouse.lower() in name.lower() or mouse in path.parts:
            mouse_id = mouse
            break

    for cond in ["A", "B", "C", "D"]:
        if f"_{cond}_" in name:
            condition = cond
            break

    for part in path.parts:
        low = part.lower()
        if low in ["condition a", "condition_a", "conditiona"]:
            condition = "A"
        elif low in ["condition b", "condition_b", "conditionb"]:
            condition = "B"
        elif low in ["condition c", "condition_c", "conditionc"]:
            condition = "C"
        elif low in ["condition d", "condition_d", "conditiond"]:
            condition = "D"

    return user_id, mouse_id, condition


def distance(a, b):
    return math.hypot(float(a["x"]) - float(b["x"]), float(a["y"]) - float(b["y"]))


def get_hit_interval_from_task_log(task_log):
    """
    Return interval from first hit click to last hit click.
    This excludes pre-task preparation and post-task adjustment.
    """
    trials = task_log.get("trials", [])
    hit_times = [
        safe_float(t.get("hit_ms"), None)
        for t in trials
        if t.get("hit_ms") is not None
    ]
    hit_times = [t for t in hit_times if t is not None]

    if len(hit_times) < 2:
        return None, None

    return min(hit_times), max(hit_times)


def filter_track_by_interval(task_log, start_ms, end_ms):
    track = task_log.get("session_mouse_track", [])
    return [
        p for p in track
        if start_ms <= safe_float(p.get("t_ms"), -1) <= end_ms
    ]


def compute_track_metrics(track, duration_ms):
    if len(track) < 2:
        return {
            "path_length_px": 0.0,
            "mean_speed_px_s": 0.0,
            "peak_speed_px_s": 0.0,
            "speed_std_px_s": 0.0,
            "movement_smoothness_proxy": 0.0,
        }

    path_len = 0.0
    speeds = []

    for i in range(1, len(track)):
        p0, p1 = track[i - 1], track[i]
        dt_ms = safe_float(p1.get("t_ms"), 0.0) - safe_float(p0.get("t_ms"), 0.0)
        d = distance(p0, p1)
        path_len += d

        if dt_ms > 1e-6:
            speeds.append(d / (dt_ms / 1000.0))

    if speeds:
        mean_speed = sum(speeds) / len(speeds)
        peak_speed = max(speeds)
        speed_std = (sum((s - mean_speed) ** 2 for s in speeds) / len(speeds)) ** 0.5
    else:
        mean_speed = peak_speed = speed_std = 0.0

    duration_s = max(duration_ms / 1000.0, 1e-6)

    # Lower is generally smoother/stabler. We keep positive direction:
    # smoothness_score = 1 / (1 + speed_std)
    smoothness_score = 1.0 / (1.0 + speed_std)

    return {
        "path_length_px": float(path_len),
        "mean_speed_px_s": float(mean_speed),
        "peak_speed_px_s": float(peak_speed),
        "speed_std_px_s": float(speed_std),
        "movement_smoothness_proxy": float(smoothness_score),
        "path_length_per_second": float(path_len / duration_s),
    }


def compute_task_log_performance(task_log_path):
    task_log = load_json(task_log_path)
    user_id, mouse_id, condition = infer_meta_from_path(task_log_path)

    start_ms, end_ms = get_hit_interval_from_task_log(task_log)
    if start_ms is None or end_ms is None:
        return None

    duration_ms = max(end_ms - start_ms, 1e-6)
    trials = [
        t for t in task_log.get("trials", [])
        if t.get("hit_ms") is not None
        and start_ms <= safe_float(t.get("hit_ms"), -1) <= end_ms
    ]

    n_hits = len(trials)
    target_count = safe_float(task_log.get("target_count"), n_hits)
    completion_ratio = n_hits / target_count if target_count > 0 else 0.0

    durations = [
        safe_float(t.get("duration_ms"), 0.0)
        for t in trials
        if t.get("duration_ms") is not None
    ]
    avg_trial_duration = sum(durations) / len(durations) if durations else 0.0

    track = filter_track_by_interval(task_log, start_ms, end_ms)
    track_metrics = compute_track_metrics(track, duration_ms)

    return {
        "source_task_log": str(task_log_path),
        "user_id": user_id,
        "mouse_id": mouse_id,
        "condition": condition,
        "interval_start_ms": float(start_ms),
        "interval_end_ms": float(end_ms),
        "active_duration_ms": float(duration_ms),
        "active_duration_s": float(duration_ms / 1000.0),
        "n_hits": int(n_hits),
        "target_count": int(target_count),
        "completion_ratio": float(completion_ratio),
        "avg_trial_duration_ms": float(avg_trial_duration),
        **track_metrics,
    }


def load_recommendations(report_path):
    if not report_path or not Path(report_path).exists():
        return {}

    report = load_json(report_path)
    user_summary = report.get("user_summary", {})
    out = {}

    for user_id, item in user_summary.items():
        rec = item.get("recommendation_placeholder", {})
        out[user_id] = {
            "recommended_mouse": rec.get("recommended_mouse", "unknown"),
            "tendency": item.get("tendency", {}).get("description", "unknown"),
            "mean_probability": item.get("mean_probability", {}),
        }

    return out


def performance_score(item):
    """
    Lower time and lower path length are better.
    Completion ratio is expected to be 1.0, but included as a safety term.

    This score is a simple normalized-free proxy:
    lower is better.
    """
    duration = safe_float(item.get("active_duration_s"), 999.0)
    path_len = safe_float(item.get("path_length_px"), 999999.0)
    completion_penalty = (1.0 - safe_float(item.get("completion_ratio"), 1.0)) * 10.0

    # Weighted simple score. You can adjust later.
    return duration + 0.001 * path_len + completion_penalty


def rank_mice_for_user(items):
    ranked = []
    for item in items:
        score = performance_score(item)
        x = dict(item)
        x["performance_score_lower_is_better"] = float(score)
        ranked.append(x)

    ranked.sort(key=lambda x: x["performance_score_lower_is_better"])
    return ranked


def main():
    parser = argparse.ArgumentParser(
        description="Analyze free-use mouse performance from first hit click to last hit click."
    )
    parser.add_argument("--raw_root", default="test/raw", help="Folder containing free task_log.json files.")
    parser.add_argument("--prediction_report", default="data/datasets/free_prediction_report.json")
    parser.add_argument("--output", default="data/datasets/free_mouse_performance_report.json")
    args = parser.parse_args()

    task_logs = sorted(Path(args.raw_root).rglob("*_task_log.json"))

    if not task_logs:
        print("[ERROR] No *_task_log.json found under:", args.raw_root)
        return

    rows = []
    for path in task_logs:
        meta_name = path.name.lower()
        meta_path = str(path).lower()

        pass
    
        result = compute_task_log_performance(path)
        if result is not None:
            rows.append(result)

    if not rows:
        print("[ERROR] No valid free task logs found.")
        return

    recommendations = load_recommendations(args.prediction_report)

    by_user = defaultdict(list)
    for r in rows:
        by_user[r["user_id"]].append(r)

    user_summary = {}

    print("=== Free Mouse Performance Summary ===")

    for user_id, items in sorted(by_user.items()):
        ranked = rank_mice_for_user(items)
        best = ranked[0]
        rec = recommendations.get(user_id, {})
        recommended_mouse = rec.get("recommended_mouse", "unknown")
        recommendation_success = (
            recommended_mouse != "unknown"
            and recommended_mouse == best["mouse_id"]
        )

        user_summary[user_id] = {
            "recommended_mouse": recommended_mouse,
            "model_tendency": rec.get("tendency", "unknown"),
            "best_mouse_by_performance": best["mouse_id"],
            "recommendation_success": bool(recommendation_success),
            "ranking": ranked,
        }

        print()
        print(f"{user_id}:")
        print(f"  model_tendency      = {rec.get('tendency', 'unknown')}")
        print(f"  recommended_mouse   = {recommended_mouse}")
        print(f"  best_mouse_actual   = {best['mouse_id']}")
        print(f"  recommendation_ok   = {recommendation_success}")
        for item in ranked:
            print(
                f"    {item['mouse_id']}: "
                f"score={item['performance_score_lower_is_better']:.3f}, "
                f"time={item['active_duration_s']:.3f}s, "
                f"avg_trial={item['avg_trial_duration_ms']:.1f}ms, "
                f"path={item['path_length_px']:.1f}, "
                f"peak_speed={item['peak_speed_px_s']:.1f}"
            )

    valid_users = [
        s for s in user_summary.values()
        if s["recommended_mouse"] != "unknown"
    ]
    n_valid = len(valid_users)
    n_success = sum(1 for s in valid_users if s["recommendation_success"])
    rec_accuracy = n_success / n_valid if n_valid > 0 else 0.0

    report = {
        "meta": {
            "raw_root": args.raw_root,
            "prediction_report": args.prediction_report,
            "n_task_logs": len(task_logs),
            "n_free_records": len(rows),
            "performance_interval": "from first hit_ms to last hit_ms",
            "score_definition": "score = active_duration_s + 0.001 * path_length_px + completion_penalty; lower is better",
        },
        "recommendation_validation": {
            "n_users_with_recommendation": n_valid,
            "n_success": n_success,
            "recommendation_accuracy": rec_accuracy,
        },
        "records": rows,
        "user_summary": user_summary,
    }

    save_json(args.output, report)

    print()
    print("=== Recommendation Validation ===")
    print(f"Users with recommendation: {n_valid}")
    print(f"Success: {n_success}")
    print(f"Recommendation accuracy: {rec_accuracy:.3f}")
    print("Saved:", args.output)


if __name__ == "__main__":
    main()
