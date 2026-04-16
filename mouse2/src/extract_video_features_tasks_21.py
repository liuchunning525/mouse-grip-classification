import argparse
import json
import math
import os
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-8:
        return np.array([1.0, 0.0], dtype=np.float32)
    return (v / n).astype(np.float32)


# -----------------------------
# Marker detection
# -----------------------------
def detect_pink_candidates(
    image_bgr: np.ndarray,
    hsv_lower: Tuple[int, int, int] = (140, 60, 60),
    hsv_upper: Tuple[int, int, int] = (179, 255, 255),
    min_area: float = 15.0,
    max_area: float = 3000.0,
) -> Tuple[List[Tuple[float, float]], np.ndarray]:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array(hsv_lower, dtype=np.uint8)
    upper = np.array(hsv_upper, dtype=np.uint8)

    mask = cv2.inRange(hsv, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    centers: List[Tuple[float, float]] = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area or area > max_area:
            continue

        x, y, w, h = cv2.boundingRect(c)
        aspect = w / h if h > 0 else 999.0
        if aspect < 0.25 or aspect > 4.5:
            continue

        M = cv2.moments(c)
        if abs(M["m00"]) < 1e-8:
            continue

        cx = float(M["m10"] / M["m00"])
        cy = float(M["m01"] / M["m00"])
        centers.append((cx, cy))

    return centers, mask


def choose_best_three_markers(marker_centers: List[Tuple[float, float]]) -> Optional[List[Tuple[float, float]]]:
    if len(marker_centers) < 3:
        return None
    if len(marker_centers) == 3:
        return list(marker_centers)

    pts = np.array(marker_centers, dtype=np.float32)
    best = None
    best_score = 1e18

    for idxs in combinations(range(len(pts)), 3):
        sub = pts[list(idxs)]
        ys = sub[:, 1]
        xs = sub[:, 0]
        y_std = np.std(ys)
        x_span = np.max(xs) - np.min(xs)
        score = y_std - 0.03 * x_span
        if score < best_score:
            best_score = score
            best = sub

    return [tuple(map(float, p)) for p in best]


def order_three_markers(marker_centers: List[Tuple[float, float]]) -> Optional[List[Tuple[float, float]]]:
    if marker_centers is None or len(marker_centers) != 3:
        return None

    pts = np.array(marker_centers, dtype=np.float32)
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
    dist_sum = d.sum(axis=1)
    mid_idx = int(np.argmin(dist_sum))

    outer_idxs = [i for i in range(3) if i != mid_idx]
    outer_pts = pts[outer_idxs]

    if outer_pts[0, 0] <= outer_pts[1, 0]:
        left_pt = outer_pts[0]
        right_pt = outer_pts[1]
    else:
        left_pt = outer_pts[1]
        right_pt = outer_pts[0]

    middle_pt = pts[mid_idx]

    ordered = [
        (float(left_pt[0]), float(left_pt[1])),
        (float(middle_pt[0]), float(middle_pt[1])),
        (float(right_pt[0]), float(right_pt[1])),
    ]
    return ordered


def affine_from_markers(ref_markers: np.ndarray, cur_markers: np.ndarray) -> Optional[np.ndarray]:
    M, _ = cv2.estimateAffinePartial2D(ref_markers, cur_markers, method=cv2.LMEDS)
    return M


def apply_affine(points: np.ndarray, M: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    transformed = cv2.transform(pts, M)
    return transformed.reshape(-1, 2)


def build_mouse_pose_from_ref(ref_pose: Dict[str, Any], M: np.ndarray) -> Dict[str, float]:
    ref_center = np.array([[ref_pose["center_x"], ref_pose["center_y"]]], dtype=np.float32)
    cur_center = apply_affine(ref_center, M)[0]

    A = M[:, :2].astype(np.float32)
    ref_major = np.array([ref_pose["major_axis_x"], ref_pose["major_axis_y"]], dtype=np.float32)
    ref_minor = np.array([ref_pose["minor_axis_x"], ref_pose["minor_axis_y"]], dtype=np.float32)

    cur_major = normalize(A @ ref_major)
    cur_minor = normalize(A @ ref_minor)
    angle_deg = math.degrees(math.atan2(float(cur_major[1]), float(cur_major[0])))

    return {
        "center_x": float(cur_center[0]),
        "center_y": float(cur_center[1]),
        "major_axis_x": float(cur_major[0]),
        "major_axis_y": float(cur_major[1]),
        "minor_axis_x": float(cur_minor[0]),
        "minor_axis_y": float(cur_minor[1]),
        "angle_deg": float(angle_deg),
    }


# -----------------------------
# MediaPipe Tasks API hand detection
# -----------------------------
def build_hand_detector(model_path: str):
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return HandLandmarker.create_from_options(options)


def extract_hand_points(frame_bgr: np.ndarray, detector: Any) -> Tuple[Optional[Dict[int, Tuple[float, float]]], Optional[str], Optional[List[List[Tuple[float, float]]]]]:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_image)

    if not result.hand_landmarks or len(result.hand_landmarks) == 0:
        return None, None, None

    hand_landmarks = result.hand_landmarks[0]
    handedness = None
    if result.handedness and len(result.handedness) > 0 and len(result.handedness[0]) > 0:
        handedness = result.handedness[0][0].category_name

    h, w = frame_bgr.shape[:2]
    pts_xy: Dict[int, Tuple[float, float]] = {}
    for idx, lm in enumerate(hand_landmarks):
        pts_xy[idx] = (float(lm.x * w), float(lm.y * h))

    # MediaPipe hand connections for drawing
    hand_connections = [
        [(0, 1), (1, 2), (2, 3), (3, 4)],
        [(0, 5), (5, 6), (6, 7), (7, 8)],
        [(0, 9), (9, 10), (10, 11), (11, 12)],
        [(0, 13), (13, 14), (14, 15), (15, 16)],
        [(0, 17), (17, 18), (18, 19), (19, 20)],
        [(5, 9), (9, 13), (13, 17)],
    ]

    return pts_xy, handedness, hand_connections


def palm_center_from_points(pts_xy: Dict[int, Tuple[float, float]]) -> Optional[Tuple[float, float]]:
    needed = [0, 5, 9, 13, 17]
    if not all(idx in pts_xy for idx in needed):
        return None
    arr = np.array([pts_xy[idx] for idx in needed], dtype=np.float32)
    c = arr.mean(axis=0)
    return float(c[0]), float(c[1])


# -----------------------------
# Relative projection
# -----------------------------
def project_to_mouse_frame(
    point_xy: Tuple[float, float],
    mouse_pose: Dict[str, float],
    ref_pose: Dict[str, Any]
) -> Dict[str, float]:
    p = np.array(point_xy, dtype=np.float32)
    c = np.array([mouse_pose["center_x"], mouse_pose["center_y"]], dtype=np.float32)
    major = np.array([mouse_pose["major_axis_x"], mouse_pose["major_axis_y"]], dtype=np.float32)
    minor = np.array([mouse_pose["minor_axis_x"], mouse_pose["minor_axis_y"]], dtype=np.float32)

    rel = p - c
    major_px = float(np.dot(rel, major))
    minor_px = float(np.dot(rel, minor))

    length_px = max(float(ref_pose.get("mouse_length_px", 1.0)), 1e-6)
    width_px = max(float(ref_pose.get("mouse_width_px", 1.0)), 1e-6)

    return {
        "x": float(point_xy[0]),
        "y": float(point_xy[1]),
        "major_px": major_px,
        "minor_px": minor_px,
        "major_norm": major_px / length_px,
        "minor_norm": minor_px / width_px,
    }


# -----------------------------
# Task log sync
# -----------------------------
def lookup_trial(task_log: Optional[Dict[str, Any]], task_t_ms: Optional[float]) -> Optional[Dict[str, Any]]:
    if task_log is None or task_t_ms is None:
        return None
    for trial in task_log.get("trials", []):
        start_ms = float(trial.get("start_ms", 0.0))
        hit_ms = float(trial.get("hit_ms", start_ms))
        if start_ms <= task_t_ms <= hit_ms:
            return {
                "trial_index": trial.get("trial_index"),
                "target_id": trial.get("target_id"),
                "target_x": trial.get("target_x"),
                "target_y": trial.get("target_y"),
                "start_ms": start_ms,
                "hit_ms": hit_ms,
            }
    return None


def build_all_hand_points(
    pts_xy: Dict[int, Tuple[float, float]],
    mouse_pose: Optional[Dict[str, float]],
    ref_pose: Dict[str, Any]
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    all_points: Dict[str, Any] = {}
    all_relative: Dict[str, Any] = {}

    for idx in range(21):
        if idx not in pts_xy:
            continue
        p = pts_xy[idx]
        all_points[str(idx)] = {"x": float(p[0]), "y": float(p[1])}
        if mouse_pose is not None:
            all_relative[str(idx)] = project_to_mouse_frame(p, mouse_pose, ref_pose)

    palm_center = palm_center_from_points(pts_xy)
    if palm_center is not None:
        all_points["palm_center"] = {"x": float(palm_center[0]), "y": float(palm_center[1])}
        if mouse_pose is not None:
            all_relative["palm_center"] = project_to_mouse_frame(palm_center, mouse_pose, ref_pose)

    return all_points, (all_relative if mouse_pose is not None else None)


def draw_hand_21_points(
    image: np.ndarray,
    pts_xy: Dict[int, Tuple[float, float]],
    connections: Optional[List[List[Tuple[int, int]]]]
) -> None:
    if not pts_xy:
        return

    if connections:
        for chain in connections:
            for a, b in chain:
                if a in pts_xy and b in pts_xy:
                    pa = (int(round(pts_xy[a][0])), int(round(pts_xy[a][1])))
                    pb = (int(round(pts_xy[b][0])), int(round(pts_xy[b][1])))
                    cv2.line(image, pa, pb, (255, 0, 0), 2)

    for idx, (x, y) in pts_xy.items():
        cv2.circle(image, (int(round(x)), int(round(y))), 3, (0, 255, 0), -1)
        cv2.putText(image, str(idx), (int(round(x)) + 4, int(round(y)) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Extract per-frame video features using MediaPipe Tasks API with full 21 hand landmarks.")
    parser.add_argument("--video", required=True, help="Input webcam video path")
    parser.add_argument("--ref_pose", required=True, help="mouse_pose_ref.json from lock_mouse_pose_standalone.py")
    parser.add_argument("--hand_model", default="hand_landmarker.task", help="Path to hand_landmarker.task")
    parser.add_argument("--output", required=True, help="Output frame_features.json")
    parser.add_argument("--task_log", default=None, help="Optional task_log.json")
    parser.add_argument("--frame_step", type=int, default=3, help="Process every Nth frame")
    parser.add_argument("--max_frames", type=int, default=-1, help="Optional hard cap on processed frames")
    parser.add_argument("--preview_video", default=None, help="Optional debug preview video")
    args = parser.parse_args()

    ref_pose = load_json(args.ref_pose)
    if "reference_markers" not in ref_pose or len(ref_pose["reference_markers"]) != 3:
        raise RuntimeError("ref_pose must contain exactly 3 reference_markers.")

    if not os.path.exists(args.hand_model):
        raise FileNotFoundError(f"Hand model not found: {args.hand_model}")

    task_log = load_json(args.task_log) if args.task_log else None
    offset_ms = task_log.get("video_task_offset_ms") if task_log else None

    ref_markers = np.array(ref_pose["reference_markers"], dtype=np.float32)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise FileNotFoundError(f"Failed to open video: {args.video}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_ms = (frame_count / fps * 1000.0) if fps > 0 else None

    preview_writer = None
    if args.preview_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
        out_fps = max((fps / max(args.frame_step, 1)) if fps > 0 else 10.0, 1.0)
        preview_writer = cv2.VideoWriter(args.preview_video, fourcc, out_fps, (w, h))

    detector = build_hand_detector(args.hand_model)

    rows: List[Dict[str, Any]] = []
    processed = 0
    frame_id = 0
    prev_mouse_center = None
    prev_t_ms = None

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_id % max(args.frame_step, 1) != 0:
                frame_id += 1
                continue

            video_t_ms = float(cap.get(cv2.CAP_PROP_POS_MSEC))
            task_t_ms = (video_t_ms - float(offset_ms)) if offset_ms is not None else None

            candidates, marker_mask = detect_pink_candidates(frame)
            cur_markers_raw = choose_best_three_markers(candidates)
            cur_markers_ordered = order_three_markers(cur_markers_raw) if cur_markers_raw is not None else None

            marker_ok = cur_markers_ordered is not None
            affine_ok = False
            mouse_pose = None

            if marker_ok:
                cur_markers = np.array(cur_markers_ordered, dtype=np.float32)
                M = affine_from_markers(ref_markers, cur_markers)
                if M is not None:
                    mouse_pose = build_mouse_pose_from_ref(ref_pose, M)
                    affine_ok = True

            pts_xy, handedness, hand_connections = extract_hand_points(frame, detector)

            mouse_dx = None
            mouse_dy = None
            mouse_speed = None
            if mouse_pose is not None:
                cur_center = np.array([mouse_pose["center_x"], mouse_pose["center_y"]], dtype=np.float32)
                if prev_mouse_center is not None and prev_t_ms is not None:
                    dt = max((video_t_ms - prev_t_ms) / 1000.0, 1e-6)
                    delta = cur_center - prev_mouse_center
                    mouse_dx = float(delta[0])
                    mouse_dy = float(delta[1])
                    mouse_speed = float(np.linalg.norm(delta) / dt)
                prev_mouse_center = cur_center
                prev_t_ms = video_t_ms

            hand_points_out: Optional[Dict[str, Any]] = None
            rel_points: Optional[Dict[str, Any]] = None
            if pts_xy:
                hand_points_out, rel_points = build_all_hand_points(pts_xy, mouse_pose, ref_pose)

            task_trial = lookup_trial(task_log, task_t_ms)

            row: Dict[str, Any] = {
                "frame_id": frame_id,
                "video_timestamp_ms": round(video_t_ms, 3),
                "task_timestamp_ms": round(task_t_ms, 3) if task_t_ms is not None else None,
                "status": {
                    "marker_ok": marker_ok,
                    "affine_ok": affine_ok,
                    "hand_ok": pts_xy is not None,
                },
                "task_trial": task_trial,
                "markers": {
                    "detected_count": len(candidates),
                    "current_markers": [[float(x), float(y)] for x, y in cur_markers_ordered] if cur_markers_ordered else None,
                },
                "mouse": {
                    "center_x": float(mouse_pose["center_x"]) if mouse_pose else None,
                    "center_y": float(mouse_pose["center_y"]) if mouse_pose else None,
                    "angle_deg": float(mouse_pose["angle_deg"]) if mouse_pose else None,
                    "dx": mouse_dx,
                    "dy": mouse_dy,
                    "speed_px_s": mouse_speed,
                },
                "hand": {
                    "handedness": handedness,
                    "points_21": hand_points_out,
                },
                "relative_to_mouse": rel_points,
            }
            rows.append(row)
            processed += 1

            if preview_writer is not None:
                vis = frame.copy()

                if cur_markers_ordered is not None:
                    for i, p in enumerate(cur_markers_ordered):
                        x, y = int(round(p[0])), int(round(p[1]))
                        cv2.circle(vis, (x, y), 8, (0, 255, 255), -1)
                        cv2.putText(vis, ["L", "M", "R"][i], (x + 6, y - 6),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                if mouse_pose is not None:
                    cx = int(round(mouse_pose["center_x"]))
                    cy = int(round(mouse_pose["center_y"]))
                    cv2.circle(vis, (cx, cy), 5, (0, 0, 255), -1)

                    major = np.array([mouse_pose["major_axis_x"], mouse_pose["major_axis_y"]], dtype=np.float32)
                    minor = np.array([mouse_pose["minor_axis_x"], mouse_pose["minor_axis_y"]], dtype=np.float32)
                    L = float(ref_pose["mouse_length_px"]) * 0.4
                    W = float(ref_pose["mouse_width_px"]) * 0.35

                    p2 = (int(round(cx + major[0] * L)), int(round(cy + major[1] * L)))
                    q2 = (int(round(cx + minor[0] * W)), int(round(cy + minor[1] * W)))
                    cv2.line(vis, (cx, cy), p2, (255, 0, 255), 2)
                    cv2.line(vis, (cx, cy), q2, (0, 165, 255), 2)

                    cv2.putText(vis, f"mouse=({cx},{cy}) angle={mouse_pose['angle_deg']:.1f}",
                                (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                if pts_xy:
                    draw_hand_21_points(vis, pts_xy, hand_connections)

                if task_t_ms is not None:
                    cv2.putText(vis, f"task_t={task_t_ms:.1f} ms", (20, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                if task_trial is not None:
                    cv2.putText(vis, f"trial={task_trial['trial_index']} target={task_trial['target_id']}", (20, 88),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                preview_writer.write(vis)

            if args.max_frames > 0 and processed >= args.max_frames:
                break

            frame_id += 1
    finally:
        cap.release()
        if preview_writer is not None:
            preview_writer.release()
        detector.close()

    out = {
        "meta": {
            "video_path": args.video,
            "ref_pose_path": args.ref_pose,
            "task_log_path": args.task_log,
            "hand_model": args.hand_model,
            "fps": fps,
            "frame_count": frame_count,
            "duration_ms": duration_ms,
            "frame_step": args.frame_step,
            "processed_frames": processed,
            "video_task_offset_ms": offset_ms,
        },
        "frames": rows,
    }
    save_json(args.output, out)

    print("=== Extraction done ===")
    print(f"Processed frames: {processed}")
    print(f"Saved JSON: {args.output}")
    if args.preview_video:
        print(f"Saved preview video: {args.preview_video}")


if __name__ == "__main__":
    main()
