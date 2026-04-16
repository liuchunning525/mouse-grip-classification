import argparse
import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def safe_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        return None


def point_distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return float(math.hypot(a[0] - b[0], a[1] - b[1]))


def series_path_length(points: List[Tuple[float, float]]) -> float:
    if len(points) < 2:
        return 0.0
    return float(sum(point_distance(points[i - 1], points[i]) for i in range(1, len(points))))


def series_straight_distance(points: List[Tuple[float, float]]) -> float:
    if len(points) < 2:
        return 0.0
    return point_distance(points[0], points[-1])


def finite_diff_speed(points: List[Tuple[float, float]], times_ms: List[float]) -> List[float]:
    speeds: List[float] = []
    for i in range(1, len(points)):
        dt = max((times_ms[i] - times_ms[i - 1]) / 1000.0, 1e-6)
        ds = point_distance(points[i], points[i - 1])
        speeds.append(float(ds / dt))
    return speeds


def finite_diff_accel(values: List[float], times_ms: List[float]) -> List[float]:
    accels: List[float] = []
    if len(values) < 2:
        return accels
    for i in range(1, len(values)):
        dt = max((times_ms[i + 1] - times_ms[i]) / 1000.0, 1e-6)
        accels.append(float((values[i] - values[i - 1]) / dt))
    return accels


def mean_std_range(vals: List[float], prefix: str) -> Dict[str, float]:
    if not vals:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_std": 0.0,
            f"{prefix}_min": 0.0,
            f"{prefix}_max": 0.0,
            f"{prefix}_range": 0.0,
        }
    arr = np.array(vals, dtype=np.float32)
    return {
        f"{prefix}_mean": float(arr.mean()),
        f"{prefix}_std": float(arr.std()),
        f"{prefix}_min": float(arr.min()),
        f"{prefix}_max": float(arr.max()),
        f"{prefix}_range": float(arr.max() - arr.min()),
    }


def corr_safe(a: List[float], b: List[float]) -> float:
    if len(a) < 2 or len(b) < 2 or len(a) != len(b):
        return 0.0
    aa = np.array(a, dtype=np.float32)
    bb = np.array(b, dtype=np.float32)
    if float(aa.std()) < 1e-8 or float(bb.std()) < 1e-8:
        return 0.0
    return float(np.corrcoef(aa, bb)[0, 1])


def get_point(frame: Dict[str, Any], key: str) -> Optional[Tuple[float, float]]:
    hand = frame.get("hand", {})
    pts = hand.get("points_21") if hand else None
    if not pts or key not in pts or pts[key] is None:
        return None
    x = safe_float(pts[key].get("x"))
    y = safe_float(pts[key].get("y"))
    if x is None or y is None:
        return None
    return (x, y)


def get_rel(frame: Dict[str, Any], key: str) -> Optional[Dict[str, float]]:
    rel = frame.get("relative_to_mouse")
    if not rel or key not in rel or rel[key] is None:
        return None
    out = {}
    for name in ["major_px", "minor_px", "major_norm", "minor_norm", "x", "y"]:
        out[name] = safe_float(rel[key].get(name))
    return out


def get_mouse_point(frame: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    mouse = frame.get("mouse", {})
    x = safe_float(mouse.get("center_x"))
    y = safe_float(mouse.get("center_y"))
    if x is None or y is None:
        return None
    return (x, y)


def get_mouse_angle(frame: Dict[str, Any]) -> Optional[float]:
    return safe_float(frame.get("mouse", {}).get("angle_deg"))


def extract_pre_hit_window(frames: List[Dict[str, Any]], hit_ms: float, window_ms: float) -> List[Dict[str, Any]]:
    start_ms = hit_ms - window_ms
    return [
        fr for fr in frames
        if (safe_float(fr.get("task_timestamp_ms")) is not None and
            start_ms <= float(fr.get("task_timestamp_ms")) <= hit_ms)
    ]


def compute_point_motion(frames: List[Dict[str, Any]], key: str, use_relative: bool = False) -> float:
    pts: List[Tuple[float, float]] = []
    for fr in frames:
        if use_relative:
            rel = get_rel(fr, key)
            if rel is None:
                continue
            mx = rel.get("major_norm")
            my = rel.get("minor_norm")
            if mx is None or my is None:
                continue
            pts.append((mx, my))
        else:
            p = get_point(fr, key)
            if p is not None:
                pts.append(p)
    return series_path_length(pts)


def compute_reach_series(frames: List[Dict[str, Any]], tip_key: str, palm_key: str = "palm_center") -> Tuple[List[float], List[float]]:
    reach_px: List[float] = []
    reach_norm: List[float] = []
    for fr in frames:
        tip = get_point(fr, tip_key)
        palm = get_point(fr, palm_key)
        wrist = get_point(fr, "0")
        middle_mcp = get_point(fr, "9")
        if tip is None or palm is None:
            continue
        d = point_distance(tip, palm)
        reach_px.append(d)
        if wrist is not None and middle_mcp is not None:
            hand_scale = max(point_distance(wrist, middle_mcp), 1e-6)
            reach_norm.append(float(d / hand_scale))
        else:
            reach_norm.append(float(d))
    return reach_px, reach_norm


def aggregate_trial(
    frames: List[Dict[str, Any]],
    trial_meta: Dict[str, Any],
    pre_hit_window_ms: float,
    angle_range_thresh: float,
    marker_ok_thresh: float,
    affine_ok_thresh: float,
    hand_ok_thresh: float,
) -> Dict[str, Any]:
    trial_frames = []
    for fr in frames:
        info = fr.get("task_trial")
        if info and info.get("trial_index") == trial_meta.get("trial_index"):
            trial_frames.append(fr)

    times_ms = [safe_float(fr.get("task_timestamp_ms")) for fr in trial_frames]
    valid = [(fr, t) for fr, t in zip(trial_frames, times_ms) if t is not None]
    trial_frames = [x[0] for x in valid]
    times_ms = [float(x[1]) for x in valid]

    mouse_points = [get_mouse_point(fr) for fr in trial_frames]
    mouse_valid = [(p, t) for p, t in zip(mouse_points, times_ms) if p is not None]
    mouse_points = [x[0] for x in mouse_valid]
    mouse_times = [x[1] for x in mouse_valid]

    features: Dict[str, float] = {}

    start_ms = safe_float(trial_meta.get("start_ms")) or 0.0
    hit_ms = safe_float(trial_meta.get("hit_ms")) or start_ms
    duration_ms = safe_float(trial_meta.get("duration_ms"))
    if duration_ms is None:
        duration_ms = max(hit_ms - start_ms, 0.0)
    features["trial_duration_ms"] = float(duration_ms)

    mouse_path = series_path_length(mouse_points)
    mouse_straight = series_straight_distance(mouse_points)
    mouse_speeds = finite_diff_speed(mouse_points, mouse_times)
    mouse_accels = finite_diff_accel(mouse_speeds, mouse_times) if len(mouse_times) >= 3 else []

    features["mouse_path_length"] = mouse_path
    features["mouse_straight_distance"] = mouse_straight
    features["mouse_path_efficiency"] = float(mouse_straight / mouse_path) if mouse_path > 1e-6 else 0.0
    features["mouse_mean_speed"] = float(np.mean(mouse_speeds)) if mouse_speeds else 0.0
    features["mouse_peak_speed"] = float(np.max(mouse_speeds)) if mouse_speeds else 0.0
    features["mouse_mean_acceleration"] = float(np.mean(np.abs(mouse_accels))) if mouse_accels else 0.0
    features["mouse_peak_acceleration"] = float(np.max(np.abs(mouse_accels))) if mouse_accels else 0.0

    if mouse_points:
        target = (
            safe_float(trial_meta.get("target_x")) or mouse_points[-1][0],
            safe_float(trial_meta.get("target_y")) or mouse_points[-1][1],
        )
        dists_to_target = [point_distance(p, target) for p in mouse_points]
        features["mouse_overshoot"] = float(max(dists_to_target) - dists_to_target[-1]) if dists_to_target else 0.0
    else:
        features["mouse_overshoot"] = 0.0

    pre_hit_frames = extract_pre_hit_window(trial_frames, hit_ms, pre_hit_window_ms)
    pre_hit_mouse = [get_mouse_point(fr) for fr in pre_hit_frames]
    pre_hit_mouse = [p for p in pre_hit_mouse if p is not None]
    features["pre_hit_mouse_jitter"] = series_path_length(pre_hit_mouse)
    features["mouse_end_jitter"] = features["pre_hit_mouse_jitter"]

    def add_keypoint_features(name: str, key: str):
        pts_valid = [p for p in (get_point(fr, key) for fr in trial_frames) if p is not None]
        rels_valid = [
            r for r in (get_rel(fr, key) for fr in trial_frames)
            if r is not None and r.get("major_norm") is not None and r.get("minor_norm") is not None
        ]
        path_len = series_path_length(pts_valid)
        features[f"{name}_path_length"] = path_len
        features[f"{name}_mouse_path_ratio"] = float(path_len / mouse_path) if mouse_path > 1e-6 else 0.0
        features.update(mean_std_range([float(r["major_norm"]) for r in rels_valid], f"{name}_rel_major"))
        features.update(mean_std_range([float(r["minor_norm"]) for r in rels_valid], f"{name}_rel_minor"))

    add_keypoint_features("wrist", "0")
    add_keypoint_features("thumb_tip", "4")
    add_keypoint_features("index_tip", "8")
    add_keypoint_features("middle_tip", "12")
    add_keypoint_features("ring_tip", "16")
    add_keypoint_features("pinky_tip", "20")
    add_keypoint_features("palm", "palm_center")

    for name, key in [("thumb", "4"), ("index", "8"), ("middle", "12"), ("ring", "16"), ("pinky", "20")]:
        reach_px, reach_norm = compute_reach_series(trial_frames, key, "palm_center")
        features.update(mean_std_range(reach_px, f"finger_{name}_reach"))
        features.update(mean_std_range(reach_norm, f"finger_{name}_reach_norm"))

    wrist_pts = [get_point(fr, "0") for fr in trial_frames]
    palm_pts = [get_point(fr, "palm_center") for fr in trial_frames]
    mouse_pts_full = [get_mouse_point(fr) for fr in trial_frames]

    def paired_xy_corr(a_pts, b_pts, prefix: str):
        paired = [(a, b) for a, b in zip(a_pts, b_pts) if a is not None and b is not None]
        if not paired:
            features[f"{prefix}_mouse_corr_x"] = 0.0
            features[f"{prefix}_mouse_corr_y"] = 0.0
            features[f"{prefix}_mouse_corr_mean"] = 0.0
            return
        ax = [p[0][0] for p in paired]
        ay = [p[0][1] for p in paired]
        bx = [p[1][0] for p in paired]
        by = [p[1][1] for p in paired]
        cx = corr_safe(ax, bx)
        cy = corr_safe(ay, by)
        features[f"{prefix}_mouse_corr_x"] = cx
        features[f"{prefix}_mouse_corr_y"] = cy
        features[f"{prefix}_mouse_corr_mean"] = float((cx + cy) * 0.5)

    paired_xy_corr(wrist_pts, mouse_pts_full, "wrist")
    paired_xy_corr(palm_pts, mouse_pts_full, "palm")

    finger_keys = ["8", "12", "16", "20"]
    finger_motion_vals = []
    name_map = {"8": "index_tip", "12": "middle_tip", "16": "ring_tip", "20": "pinky_tip"}
    for k in finger_keys:
        m = compute_point_motion(trial_frames, k, use_relative=True)
        finger_motion_vals.append(m)
        features[f"{name_map[k]}_relative_motion"] = m

    features["fingertip_rel_motion_mean"] = float(np.mean(finger_motion_vals)) if finger_motion_vals else 0.0
    features["fingertip_rel_motion_peak"] = float(np.max(finger_motion_vals)) if finger_motion_vals else 0.0
    features["finger_motion_ratio"] = float(features["fingertip_rel_motion_mean"] / mouse_path) if mouse_path > 1e-6 else 0.0

    features["pre_hit_index_motion"] = compute_point_motion(pre_hit_frames, "8", use_relative=True)
    features["pre_hit_middle_motion"] = compute_point_motion(pre_hit_frames, "12", use_relative=True)
    features["pre_hit_thumb_motion"] = compute_point_motion(pre_hit_frames, "4", use_relative=True)
    features["pre_hit_pinky_motion"] = compute_point_motion(pre_hit_frames, "20", use_relative=True)

    angle_vals = [get_mouse_angle(fr) for fr in trial_frames]
    angle_vals = [v for v in angle_vals if v is not None]
    features.update(mean_std_range(angle_vals, "mouse_angle"))

    n_frames = len(trial_frames)
    marker_ok_count = sum(1 for fr in trial_frames if fr.get("status", {}).get("marker_ok"))
    affine_ok_count = sum(1 for fr in trial_frames if fr.get("status", {}).get("affine_ok"))
    hand_ok_count = sum(1 for fr in trial_frames if fr.get("status", {}).get("hand_ok"))

    marker_ok_ratio = float(marker_ok_count / n_frames) if n_frames else 0.0
    affine_ok_ratio = float(affine_ok_count / n_frames) if n_frames else 0.0
    hand_ok_ratio = float(hand_ok_count / n_frames) if n_frames else 0.0

    features["n_frames"] = float(n_frames)
    features["marker_ok_ratio"] = marker_ok_ratio
    features["affine_ok_ratio"] = affine_ok_ratio
    features["hand_ok_ratio"] = hand_ok_ratio

    angle_range = features.get("mouse_angle_range", 0.0)
    pose_stable = (
        marker_ok_ratio >= marker_ok_thresh and
        affine_ok_ratio >= affine_ok_thresh and
        hand_ok_ratio >= hand_ok_thresh and
        angle_range <= angle_range_thresh
    )
    features["pose_stable"] = 1.0 if pose_stable else 0.0

    return {
        "trial_index": trial_meta.get("trial_index"),
        "target_id": trial_meta.get("target_id"),
        "target_x": trial_meta.get("target_x"),
        "target_y": trial_meta.get("target_y"),
        "start_ms": start_ms,
        "hit_ms": hit_ms,
        "duration_ms": duration_ms,
        "features": features,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate upgraded trial-level features with finger reach and pose stability.")
    parser.add_argument("--frame_features", required=True, help="Path to frame_features.json")
    parser.add_argument("--task_log", required=True, help="Path to task_log.json")
    parser.add_argument("--output", required=True, help="Path to output trial_features.json")
    parser.add_argument("--pre_hit_window_ms", type=float, default=250.0)
    parser.add_argument("--angle_range_thresh", type=float, default=20.0)
    parser.add_argument("--marker_ok_thresh", type=float, default=0.95)
    parser.add_argument("--affine_ok_thresh", type=float, default=0.95)
    parser.add_argument("--hand_ok_thresh", type=float, default=0.95)
    args = parser.parse_args()

    frame_data = load_json(args.frame_features)
    task_log = load_json(args.task_log)

    frames = frame_data.get("frames", [])
    trials = task_log.get("trials", [])

    out_trials = [
        aggregate_trial(
            frames,
            trial,
            args.pre_hit_window_ms,
            args.angle_range_thresh,
            args.marker_ok_thresh,
            args.affine_ok_thresh,
            args.hand_ok_thresh,
        )
        for trial in trials
    ]

    output = {
        "meta": {
            "frame_features_path": args.frame_features,
            "task_log_path": args.task_log,
            "pre_hit_window_ms": args.pre_hit_window_ms,
            "angle_range_thresh": args.angle_range_thresh,
            "marker_ok_thresh": args.marker_ok_thresh,
            "affine_ok_thresh": args.affine_ok_thresh,
            "hand_ok_thresh": args.hand_ok_thresh,
            "n_trials": len(out_trials),
            "source_meta": frame_data.get("meta", {}),
        },
        "trials": out_trials,
    }
    save_json(args.output, output)

    print("=== Upgraded trial aggregation done ===")
    print(f"Trials: {len(out_trials)}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
