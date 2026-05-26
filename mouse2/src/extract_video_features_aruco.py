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
# ArUco marker detection
# -----------------------------
def get_aruco_dict(name: str):
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("Your OpenCV does not include aruco. Install opencv-contrib-python.")
    if not hasattr(cv2.aruco, name):
        raise ValueError(f"Unknown ArUco dictionary: {name}")
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))


def create_aruco_detector(dictionary_name: str):
    aruco_dict = get_aruco_dict(dictionary_name)

    if hasattr(cv2.aruco, "ArucoDetector"):
        params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, params)

        def detect(gray):
            return detector.detectMarkers(gray)

        return detect

    params = cv2.aruco.DetectorParameters_create()

    def detect(gray):
        return cv2.aruco.detectMarkers(gray, aruco_dict, parameters=params)

    return detect


def detect_one_aruco(
    frame_bgr: np.ndarray,
    detect_fn: Any,
    marker_id: int
) -> Optional[Dict[str, Any]]:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    corners_list, ids, rejected = detect_fn(gray)

    if ids is None:
        return None

    ids_flat = ids.flatten().tolist()
    for i, mid in enumerate(ids_flat):
        if int(mid) == int(marker_id):
            corners = corners_list[i].reshape(4, 2).astype(np.float32)
            center = corners.mean(axis=0)
            v = corners[1] - corners[0]
            angle_deg = math.degrees(math.atan2(float(v[1]), float(v[0])))
            side_px = float((
                np.linalg.norm(corners[1] - corners[0])
                + np.linalg.norm(corners[2] - corners[1])
                + np.linalg.norm(corners[3] - corners[2])
                + np.linalg.norm(corners[0] - corners[3])
            ) / 4.0)

            return {
                "id": int(mid),
                "corners": corners,
                "center_x": float(center[0]),
                "center_y": float(center[1]),
                "angle_deg": float(angle_deg),
                "side_px": side_px,
                "detected_count": len(ids_flat),
            }

    return None


def unit_from_angle(angle_deg: float) -> Tuple[np.ndarray, np.ndarray]:
    theta = math.radians(angle_deg)
    major = np.array([math.cos(theta), math.sin(theta)], dtype=np.float32)
    major = normalize(major)
    minor = np.array([-major[1], major[0]], dtype=np.float32)
    return major, minor


def build_mouse_pose_from_aruco(ref_pose: Dict[str, Any], aruco: Dict[str, Any]) -> Dict[str, float]:
    major, minor = unit_from_angle(float(aruco["angle_deg"]))

    return {
        "center_x": float(aruco["center_x"]),
        "center_y": float(aruco["center_y"]),
        "major_axis_x": float(major[0]),
        "major_axis_y": float(major[1]),
        "minor_axis_x": float(minor[0]),
        "minor_axis_y": float(minor[1]),
        "angle_deg": float(aruco["angle_deg"]),
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


def extract_hand_points(frame_bgr: np.ndarray, detector: Any) -> Tuple[Optional[Dict[int, Tuple[float, float, float]]], Optional[str], Optional[List[List[Tuple[float, float]]]]]:
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
    pts_xy: Dict[int, Tuple[float, float, float]] = {}
    for idx, lm in enumerate(hand_landmarks):
        pts_xy[idx] = (float(lm.x * w), float(lm.y * h), float(lm.z))

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


def palm_center_from_points(pts_xy: Dict[int, Tuple[float, float, float]]) -> Optional[Tuple[float, float, float]]:
    needed = [0, 5, 9, 13, 17]
    if not all(idx in pts_xy for idx in needed):
        return None
    arr = np.array([pts_xy[idx] for idx in needed], dtype=np.float32)
    c = arr.mean(axis=0)
    return float(c[0]), float(c[1]), float(c[2])


# -----------------------------
# Relative projection
# -----------------------------
def project_to_mouse_frame(
    point_xy: Tuple[float, ...],
    mouse_pose: Dict[str, float],
    ref_pose: Dict[str, Any]
) -> Dict[str, float]:
    p = np.array([point_xy[0], point_xy[1]], dtype=np.float32)
    c = np.array([mouse_pose["center_x"], mouse_pose["center_y"]], dtype=np.float32)
    major = np.array([mouse_pose["major_axis_x"], mouse_pose["major_axis_y"]], dtype=np.float32)
    minor = np.array([mouse_pose["minor_axis_x"], mouse_pose["minor_axis_y"]], dtype=np.float32)

    rel = p - c
    major_px = float(np.dot(rel, major))
    minor_px = float(np.dot(rel, minor))

    length_px = max(float(ref_pose.get("mouse_length_px", 1.0)), 1e-6)
    width_px = max(float(ref_pose.get("mouse_width_px", 1.0)), 1e-6)

    out = {
        "x": float(point_xy[0]),
        "y": float(point_xy[1]),
        "major_px": major_px,
        "minor_px": minor_px,
        "major_norm": major_px / length_px,
        "minor_norm": minor_px / width_px,
    }
    if len(point_xy) >= 3:
        out["z"] = float(point_xy[2])
    return out


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
    pts_xy: Dict[int, Tuple[float, float, float]],
    mouse_pose: Optional[Dict[str, float]],
    ref_pose: Dict[str, Any]
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    all_points: Dict[str, Any] = {}
    all_relative: Dict[str, Any] = {}

    for idx in range(21):
        if idx not in pts_xy:
            continue
        p = pts_xy[idx]
        all_points[str(idx)] = {"x": float(p[0]), "y": float(p[1]), "z": float(p[2])}
        if mouse_pose is not None:
            all_relative[str(idx)] = project_to_mouse_frame(p, mouse_pose, ref_pose)

    palm_center = palm_center_from_points(pts_xy)
    if palm_center is not None:
        all_points["palm_center"] = {"x": float(palm_center[0]), "y": float(palm_center[1]), "z": float(palm_center[2])}
        if mouse_pose is not None:
            all_relative["palm_center"] = project_to_mouse_frame(palm_center, mouse_pose, ref_pose)

    return all_points, (all_relative if mouse_pose is not None else None)


def draw_hand_21_points(
    image: np.ndarray,
    pts_xy: Dict[int, Tuple[float, float, float]],
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

    for idx, p in pts_xy.items():
        x = p[0]
        y = p[1]
        cv2.circle(image, (int(round(x)), int(round(y))), 3, (0, 255, 0), -1)
        cv2.putText(image, str(idx), (int(round(x)) + 4, int(round(y)) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Extract per-frame video features using MediaPipe Tasks API + single ArUco mouse tracking.")
    parser.add_argument("--video", required=True, help="Input webcam video path")
    parser.add_argument("--ref_pose", required=True, help="mouse_pose_ref.json from lock_mouse_pose_standalone.py")
    parser.add_argument("--hand_model", default="hand_landmarker.task", help="Path to hand_landmarker.task")
    parser.add_argument("--output", required=True, help="Output frame_features.json")
    parser.add_argument("--task_log", default=None, help="Optional task_log.json")
    parser.add_argument("--frame_step", type=int, default=3, help="Process every Nth frame")
    parser.add_argument("--max_frames", type=int, default=-1, help="Optional hard cap on processed frames")
    parser.add_argument("--preview_video", default=None, help="Optional debug preview video")
    parser.add_argument("--aruco_dict", default=None, help="Override ArUco dictionary, e.g. DICT_4X4_50")
    parser.add_argument("--aruco_id", type=int, default=None, help="Override ArUco id")
    args = parser.parse_args()

    ref_pose = load_json(args.ref_pose)
    if ref_pose.get("pose_type") != "aruco_single_marker":
        raise RuntimeError("ref_pose must be created by detect_aruco_mouse_ref.py and contain pose_type='aruco_single_marker'.")

    if not os.path.exists(args.hand_model):
        raise FileNotFoundError(f"Hand model not found: {args.hand_model}")

    task_log = load_json(args.task_log) if args.task_log else None
    offset_ms = task_log.get("video_task_offset_ms") if task_log else None

    aruco_dict_name = args.aruco_dict or ref_pose.get("aruco_dictionary", "DICT_4X4_50")
    aruco_id = int(args.aruco_id if args.aruco_id is not None else ref_pose.get("aruco_id", 0))
    aruco_detect_fn = create_aruco_detector(aruco_dict_name)

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

            aruco = detect_one_aruco(frame, aruco_detect_fn, aruco_id)

            marker_ok = aruco is not None
            affine_ok = marker_ok
            mouse_pose = None

            if marker_ok:
                mouse_pose = build_mouse_pose_from_aruco(ref_pose, aruco)

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
                    "type": "aruco",
                    "dictionary": aruco_dict_name,
                    "target_id": aruco_id,
                    "detected_count": int(aruco["detected_count"]) if aruco else 0,
                    "current_markers": [[float(x), float(y)] for x, y in aruco["corners"]] if aruco else None,
                    "center": [float(aruco["center_x"]), float(aruco["center_y"])] if aruco else None,
                    "side_px": float(aruco["side_px"]) if aruco else None,
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

                if aruco is not None:
                    corners = aruco["corners"].astype(int)
                    cv2.polylines(vis, [corners], True, (0, 255, 0), 2)
                    acx = int(round(aruco["center_x"]))
                    acy = int(round(aruco["center_y"]))
                    cv2.circle(vis, (acx, acy), 6, (0, 0, 255), -1)
                    cv2.putText(vis, f"ArUco id={aruco_id}", (acx + 8, acy - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

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
            "mouse_tracking": "aruco_single_marker",
            "aruco_dictionary": aruco_dict_name,
            "aruco_id": aruco_id,
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
