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


def angle_3points_2d(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> Optional[float]:
    va = np.array([a[0] - b[0], a[1] - b[1]], dtype=np.float32)
    vc = np.array([c[0] - b[0], c[1] - b[1]], dtype=np.float32)
    na = float(np.linalg.norm(va))
    nc = float(np.linalg.norm(vc))
    if na < 1e-8 or nc < 1e-8:
        return None
    cosv = float(np.clip(np.dot(va, vc) / (na * nc), -1.0, 1.0))
    return float(math.degrees(math.acos(cosv)))


def angle_3points_3d(a, b, c, z_scale: float = 1.0) -> Optional[float]:
    va = np.array([a[0] - b[0], a[1] - b[1], (a[2] - b[2]) * z_scale], dtype=np.float32)
    vc = np.array([c[0] - b[0], c[1] - b[1], (c[2] - b[2]) * z_scale], dtype=np.float32)
    na = float(np.linalg.norm(va))
    nc = float(np.linalg.norm(vc))
    if na < 1e-8 or nc < 1e-8:
        return None
    cosv = float(np.clip(np.dot(va, vc) / (na * nc), -1.0, 1.0))
    return float(math.degrees(math.acos(cosv)))


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


def get_point3d(frame: Dict[str, Any], key: str) -> Optional[Tuple[float, float, float]]:
    hand = frame.get("hand", {})
    pts = hand.get("points_21") if hand else None
    if not pts or key not in pts or pts[key] is None:
        return None
    x = safe_float(pts[key].get("x"))
    y = safe_float(pts[key].get("y"))
    z = safe_float(pts[key].get("z"))
    if x is None or y is None:
        return None
    if z is None:
        z = 0.0
    return (x, y, z)


def get_point_z(frame: Dict[str, Any], key: str) -> Optional[float]:
    p = get_point3d(frame, key)
    return None if p is None else float(p[2])


def get_rel(frame: Dict[str, Any], key: str) -> Optional[Dict[str, float]]:
    rel = frame.get("relative_to_mouse")
    if not rel or key not in rel or rel[key] is None:
        return None
    out = {}
    for name in ["major_px", "minor_px", "major_norm", "minor_norm", "x", "y", "z"]:
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


def compute_hand_scale(frames: List[Dict[str, Any]]) -> float:
    vals = []
    for fr in frames:
        wrist = get_point(fr, "0")
        middle_mcp = get_point(fr, "9")
        if wrist is not None and middle_mcp is not None:
            vals.append(point_distance(wrist, middle_mcp))
    if not vals:
        return 1.0
    return max(float(np.median(np.array(vals, dtype=np.float32))), 1e-6)


def add_hand_pose_features(features: Dict[str, float], frames: List[Dict[str, Any]]) -> None:
    if not frames:
        return

    hand_scale = compute_hand_scale(frames)
    z_scale = hand_scale

    finger_defs = {
        "thumb":  ("1", "2", "3", "4"),
        "index":  ("5", "6", "7", "8"),
        "middle": ("9", "10", "11", "12"),
        "ring":   ("13", "14", "15", "16"),
        "pinky":  ("17", "18", "19", "20"),
    }

    all_tip_z = []
    all_mcp_z = []
    all_pip_z = []
    all_dip_z = []

    for fname, (mcp, pip, dip, tip) in finger_defs.items():
        pip_angles_2d, dip_angles_2d = [], []
        pip_angles_3d, dip_angles_3d = [], []
        tip_to_mcp_norm, tip_to_palm_norm = [], []
        curl_proxy = []
        z_tip_minus_mcp, z_tip_minus_palm = [], []
        z_pip_minus_mcp, z_dip_minus_mcp = [], []

        for fr in frames:
            mcp2, pip2, dip2, tip2 = get_point(fr, mcp), get_point(fr, pip), get_point(fr, dip), get_point(fr, tip)
            palm2 = get_point(fr, "palm_center")
            mcp3, pip3, dip3, tip3 = get_point3d(fr, mcp), get_point3d(fr, pip), get_point3d(fr, dip), get_point3d(fr, tip)
            palm3 = get_point3d(fr, "palm_center")

            if mcp2 is not None and pip2 is not None and dip2 is not None:
                a = angle_3points_2d(mcp2, pip2, dip2)
                if a is not None:
                    pip_angles_2d.append(a)
            if pip2 is not None and dip2 is not None and tip2 is not None:
                a = angle_3points_2d(pip2, dip2, tip2)
                if a is not None:
                    dip_angles_2d.append(a)

            if mcp3 is not None and pip3 is not None and dip3 is not None:
                a = angle_3points_3d(mcp3, pip3, dip3, z_scale=z_scale)
                if a is not None:
                    pip_angles_3d.append(a)
            if pip3 is not None and dip3 is not None and tip3 is not None:
                a = angle_3points_3d(pip3, dip3, tip3, z_scale=z_scale)
                if a is not None:
                    dip_angles_3d.append(a)

            if mcp2 is not None and tip2 is not None:
                tip_to_mcp_norm.append(point_distance(mcp2, tip2) / hand_scale)
            if palm2 is not None and tip2 is not None:
                tip_to_palm_norm.append(point_distance(palm2, tip2) / hand_scale)

            if mcp2 is not None and pip2 is not None and dip2 is not None and tip2 is not None:
                bone_len = point_distance(mcp2, pip2) + point_distance(pip2, dip2) + point_distance(dip2, tip2)
                direct = point_distance(mcp2, tip2)
                curl_proxy.append(float(1.0 - direct / max(bone_len, 1e-6)))

            if mcp3 is not None and tip3 is not None:
                z_tip_minus_mcp.append(float(tip3[2] - mcp3[2]))
                all_tip_z.append(float(tip3[2]))
                all_mcp_z.append(float(mcp3[2]))
            if palm3 is not None and tip3 is not None:
                z_tip_minus_palm.append(float(tip3[2] - palm3[2]))
            if mcp3 is not None and pip3 is not None:
                z_pip_minus_mcp.append(float(pip3[2] - mcp3[2]))
                all_pip_z.append(float(pip3[2]))
            if mcp3 is not None and dip3 is not None:
                z_dip_minus_mcp.append(float(dip3[2] - mcp3[2]))
                all_dip_z.append(float(dip3[2]))

        features.update(mean_std_range(pip_angles_2d, f"{fname}_pip_angle_2d"))
        features.update(mean_std_range(dip_angles_2d, f"{fname}_dip_angle_2d"))
        features.update(mean_std_range(pip_angles_3d, f"{fname}_pip_angle_3d"))
        features.update(mean_std_range(dip_angles_3d, f"{fname}_dip_angle_3d"))
        features.update(mean_std_range(tip_to_mcp_norm, f"{fname}_tip_to_mcp_norm"))
        features.update(mean_std_range(tip_to_palm_norm, f"{fname}_tip_to_palm_norm"))
        features.update(mean_std_range(curl_proxy, f"{fname}_curl_proxy"))
        features.update(mean_std_range(z_tip_minus_mcp, f"{fname}_z_tip_minus_mcp"))
        features.update(mean_std_range(z_tip_minus_palm, f"{fname}_z_tip_minus_palm"))
        features.update(mean_std_range(z_pip_minus_mcp, f"{fname}_z_pip_minus_mcp"))
        features.update(mean_std_range(z_dip_minus_mcp, f"{fname}_z_dip_minus_mcp"))

    main_fingers = ["index", "middle", "ring", "pinky"]
    for stat in ["mean", "std", "min", "max", "range"]:
        features[f"main_finger_curl_proxy_{stat}_mean"] = float(np.mean([features.get(f"{f}_curl_proxy_{stat}", 0.0) for f in main_fingers]))
        features[f"main_finger_extension_{stat}_mean"] = float(np.mean([features.get(f"{f}_tip_to_mcp_norm_{stat}", 0.0) for f in main_fingers]))
        features[f"main_finger_pip_angle_3d_{stat}_mean"] = float(np.mean([features.get(f"{f}_pip_angle_3d_{stat}", 0.0) for f in main_fingers]))

    spread_pairs = [
        ("thumb_index", "4", "8"),
        ("index_middle", "8", "12"),
        ("middle_ring", "12", "16"),
        ("ring_pinky", "16", "20"),
        ("index_pinky", "8", "20"),
    ]
    all_spreads = []
    for pname, akey, bkey in spread_pairs:
        vals = []
        for fr in frames:
            a, b = get_point(fr, akey), get_point(fr, bkey)
            if a is not None and b is not None:
                v = point_distance(a, b) / hand_scale
                vals.append(v)
                all_spreads.append(v)
        features.update(mean_std_range(vals, f"finger_spread_{pname}_norm"))
    features.update(mean_std_range(all_spreads, "finger_spread_all_norm"))

    compact_vals = []
    for fr in frames:
        palm = get_point(fr, "palm_center")
        if palm is None:
            continue
        dists = []
        for tip in ["4", "8", "12", "16", "20"]:
            p = get_point(fr, tip)
            if p is not None:
                dists.append(point_distance(palm, p) / hand_scale)
        if dists:
            compact_vals.append(float(np.mean(dists)))
    features.update(mean_std_range(compact_vals, "hand_compactness_tip_to_palm_norm"))

    z_name_map = {
        "0": "wrist", "5": "index_mcp", "9": "middle_mcp",
        "13": "ring_mcp", "17": "pinky_mcp",
        "4": "thumb_tip", "8": "index_tip", "12": "middle_tip",
        "16": "ring_tip", "20": "pinky_tip", "palm_center": "palm_center",
    }
    for key, name in z_name_map.items():
        vals = [get_point_z(fr, key) for fr in frames]
        vals = [v for v in vals if v is not None]
        features.update(mean_std_range(vals, f"{name}_z"))

    features.update(mean_std_range(all_tip_z, "all_tips_z"))
    features.update(mean_std_range(all_mcp_z, "all_mcps_z"))
    features.update(mean_std_range(all_pip_z, "all_pips_z"))
    features.update(mean_std_range(all_dip_z, "all_dips_z"))

    features["tip_mcp_z_mean_diff"] = float(np.mean(all_tip_z) - np.mean(all_mcp_z)) if all_tip_z and all_mcp_z else 0.0
    features["tip_z_variance"] = float(np.var(np.array(all_tip_z, dtype=np.float32))) if all_tip_z else 0.0
    features["hand_scale_px"] = float(hand_scale)



def add_static_curl_and_motion_ratio_features(features: Dict[str, float], frames: List[Dict[str, Any]]) -> None:
    """
    Add features for:
    1) Claw: static curled posture
       - fingers stay curled
       - curl mean high
       - tip-to-MCP extension low
       - PIP/DIP angle mean lower
       - curl std low

    2) Fingertip: dynamic finger control
       - fingertip relative motion large
       - wrist/palm relative motion small
       - finger motion / wrist motion ratio high
       - fingertip extension changes more
    """
    if not frames:
        return

    main_fingers = ["index", "middle", "ring", "pinky"]

    # -----------------------------
    # Static claw posture score
    # -----------------------------
    curl_means = [features.get(f"{f}_curl_proxy_mean", 0.0) for f in main_fingers]
    curl_stds = [features.get(f"{f}_curl_proxy_std", 0.0) for f in main_fingers]
    extension_means = [features.get(f"{f}_tip_to_mcp_norm_mean", 0.0) for f in main_fingers]
    extension_stds = [features.get(f"{f}_tip_to_mcp_norm_std", 0.0) for f in main_fingers]

    pip_angle_means = [features.get(f"{f}_pip_angle_3d_mean", 0.0) for f in main_fingers]
    dip_angle_means = [features.get(f"{f}_dip_angle_3d_mean", 0.0) for f in main_fingers]
    pip_angle_stds = [features.get(f"{f}_pip_angle_3d_std", 0.0) for f in main_fingers]
    dip_angle_stds = [features.get(f"{f}_dip_angle_3d_std", 0.0) for f in main_fingers]

    features["static_curl_mean_main"] = float(np.mean(curl_means)) if curl_means else 0.0
    features["static_curl_std_main"] = float(np.mean(curl_stds)) if curl_stds else 0.0
    features["static_extension_mean_main"] = float(np.mean(extension_means)) if extension_means else 0.0
    features["static_extension_std_main"] = float(np.mean(extension_stds)) if extension_stds else 0.0

    features["static_pip_angle_mean_main"] = float(np.mean(pip_angle_means)) if pip_angle_means else 0.0
    features["static_dip_angle_mean_main"] = float(np.mean(dip_angle_means)) if dip_angle_means else 0.0
    features["static_joint_angle_mean_main"] = float(np.mean(pip_angle_means + dip_angle_means)) if (pip_angle_means or dip_angle_means) else 0.0
    features["static_joint_angle_std_main"] = float(np.mean(pip_angle_stds + dip_angle_stds)) if (pip_angle_stds or dip_angle_stds) else 0.0

    # Higher means more claw-like:
    # curled posture high, extension low, and curl is stable.
    features["claw_static_posture_score"] = (
        features["static_curl_mean_main"]
        - 0.35 * features["static_extension_mean_main"]
        - 0.50 * features["static_curl_std_main"]
        - 0.15 * features["static_extension_std_main"]
    )

    # Lower joint angle usually means stronger bend.
    # Convert it into a bend score.
    features["claw_joint_bend_score"] = (
        (180.0 - features["static_joint_angle_mean_main"]) / 180.0
        - 0.25 * features["static_joint_angle_std_main"] / 180.0
    )

    # -----------------------------
    # Fingertip dynamic-control score
    # -----------------------------
    # Existing relative motion features are normalized movement in mouse/marker frame.
    finger_motion_vals = [
        features.get("index_tip_relative_motion", 0.0),
        features.get("middle_tip_relative_motion", 0.0),
        features.get("ring_tip_relative_motion", 0.0),
        features.get("pinky_tip_relative_motion", 0.0),
    ]
    finger_motion_mean = float(np.mean(finger_motion_vals)) if finger_motion_vals else 0.0
    finger_motion_max = float(np.max(finger_motion_vals)) if finger_motion_vals else 0.0

    wrist_motion = compute_point_motion(frames, "0", use_relative=True)
    palm_motion = compute_point_motion(frames, "palm_center", use_relative=True)
    thumb_motion = compute_point_motion(frames, "4", use_relative=True)

    features["wrist_relative_motion"] = float(wrist_motion)
    features["palm_relative_motion"] = float(palm_motion)
    features["thumb_relative_motion"] = float(thumb_motion)

    features["finger_to_wrist_motion_ratio"] = float(finger_motion_mean / (wrist_motion + 1e-6))
    features["finger_to_palm_motion_ratio"] = float(finger_motion_mean / (palm_motion + 1e-6))
    features["finger_to_handbase_motion_ratio"] = float(finger_motion_mean / ((wrist_motion + palm_motion) * 0.5 + 1e-6))
    features["thumb_index_to_wrist_motion_ratio"] = float((thumb_motion + features.get("index_tip_relative_motion", 0.0)) * 0.5 / (wrist_motion + 1e-6))
    features["finger_motion_minus_wrist_motion"] = float(finger_motion_mean - wrist_motion)
    features["finger_motion_minus_palm_motion"] = float(finger_motion_mean - palm_motion)

    # Fingertip should have dynamic finger control:
    # finger motion high, wrist/palm motion relatively low.
    features["fingertip_dynamic_control_score"] = (
        finger_motion_mean
        + 0.5 * finger_motion_max
        - 0.5 * wrist_motion
        - 0.3 * palm_motion
    )

    # Extension change: fingertip often has more finger push-pull variation.
    extension_std_vals = [features.get(f"{f}_tip_to_mcp_norm_std", 0.0) for f in ["thumb", "index", "middle", "ring", "pinky"]]
    tip_to_palm_std_vals = [features.get(f"{f}_tip_to_palm_norm_std", 0.0) for f in ["thumb", "index", "middle", "ring", "pinky"]]

    features["finger_extension_change_mean"] = float(np.mean(extension_std_vals)) if extension_std_vals else 0.0
    features["finger_tip_to_palm_change_mean"] = float(np.mean(tip_to_palm_std_vals)) if tip_to_palm_std_vals else 0.0

    features["fingertip_push_pull_score"] = (
        features["finger_extension_change_mean"]
        + features["finger_tip_to_palm_change_mean"]
        + 0.5 * features["finger_to_wrist_motion_ratio"]
    )

    # -----------------------------
    # Palm support / handbase motion
    # -----------------------------
    # Palm grip tends to move as a whole hand: palm/wrist motion follows marker/mouse.
    features["handbase_relative_motion_mean"] = float((wrist_motion + palm_motion) * 0.5)
    features["finger_vs_handbase_motion_gap"] = float(finger_motion_mean - features["handbase_relative_motion_mean"])


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

    # New intrinsic hand-pose features: curl, extension, spread, and z-pattern.
    add_hand_pose_features(features, trial_frames)

    # Extra features for claw-vs-fingertip:
    # claw = static curled posture; fingertip = dynamic finger control with stable wrist/palm.
    add_static_curl_and_motion_ratio_features(features, trial_frames)

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

    pose_stable = (
        marker_ok_ratio >= 0.75 and
        hand_ok_ratio >= 0.75
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
