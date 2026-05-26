import argparse
import json
import math
import os

import cv2
import numpy as np


def save_json(path, data):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def get_aruco_dict(name):
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("Your OpenCV does not include aruco. Install opencv-contrib-python.")
    if not hasattr(cv2.aruco, name):
        raise ValueError(f"Unknown ArUco dictionary: {name}")
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))


def create_detector(dictionary_name):
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


def detect_one_aruco(image_bgr, dictionary_name="DICT_4X4_50", marker_id=0):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    detect = create_detector(dictionary_name)
    corners_list, ids, rejected = detect(gray)

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
            }

    return None


def unit_from_angle(angle_deg):
    theta = math.radians(angle_deg)
    major = np.array([math.cos(theta), math.sin(theta)], dtype=np.float32)
    major /= max(float(np.linalg.norm(major)), 1e-8)
    minor = np.array([-major[1], major[0]], dtype=np.float32)
    return major, minor


def draw_vis(image, aruco, pose):
    vis = image.copy()
    corners = aruco["corners"].astype(int)
    cv2.polylines(vis, [corners], True, (0, 255, 0), 3)

    cx = int(round(aruco["center_x"]))
    cy = int(round(aruco["center_y"]))
    cv2.circle(vis, (cx, cy), 6, (0, 0, 255), -1)

    major = np.array([pose["major_axis_x"], pose["major_axis_y"]], dtype=np.float32)
    minor = np.array([pose["minor_axis_x"], pose["minor_axis_y"]], dtype=np.float32)

    L = max(float(pose.get("mouse_length_px", aruco["side_px"] * 4.0)), 80.0) * 0.25
    W = max(float(pose.get("mouse_width_px", aruco["side_px"] * 2.0)), 40.0) * 0.25

    p2 = (int(round(cx + major[0] * L)), int(round(cy + major[1] * L)))
    q2 = (int(round(cx + minor[0] * W)), int(round(cy + minor[1] * W)))

    cv2.arrowedLine(vis, (cx, cy), p2, (255, 0, 255), 3)
    cv2.arrowedLine(vis, (cx, cy), q2, (0, 165, 255), 3)

    cv2.putText(vis, f"ArUco id={aruco['id']} angle={aruco['angle_deg']:.1f}", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(vis, f"center=({cx},{cy}) side={aruco['side_px']:.1f}px", (20, 68),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    return vis


def main():
    parser = argparse.ArgumentParser(description="Create mouse reference pose using one ArUco marker.")
    parser.add_argument("--image", required=True, help="Reference image with ArUco marker on mouse")
    parser.add_argument("--output_dir", default="data/outputs/ref")
    parser.add_argument("--output_name", default="aruco_mouse_pose_ref.json")
    parser.add_argument("--dict", default="DICT_4X4_50")
    parser.add_argument("--id", type=int, default=0)
    parser.add_argument("--real_length_mm", type=float, default=None)
    parser.add_argument("--real_width_mm", type=float, default=None)
    parser.add_argument("--marker_size_mm", type=float, default=None, help="Printed black square size in mm")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {args.image}")

    aruco = detect_one_aruco(image, args.dict, args.id)
    if aruco is None:
        raise RuntimeError(f"Failed to detect ArUco id={args.id} in image: {args.image}")

    major, minor = unit_from_angle(aruco["angle_deg"])

    mm_per_pixel_marker = None
    if args.marker_size_mm is not None and aruco["side_px"] > 1e-8:
        mm_per_pixel_marker = float(args.marker_size_mm / aruco["side_px"])

    pose = {
        "pose_type": "aruco_single_marker",
        "source_image": os.path.basename(args.image),
        "aruco_dictionary": args.dict,
        "aruco_id": int(args.id),

        "center_x": float(aruco["center_x"]),
        "center_y": float(aruco["center_y"]),

        "major_axis_x": float(major[0]),
        "major_axis_y": float(major[1]),
        "minor_axis_x": float(minor[0]),
        "minor_axis_y": float(minor[1]),
        "angle_deg": float(aruco["angle_deg"]),

        "marker_side_px": float(aruco["side_px"]),
        "marker_size_mm": float(args.marker_size_mm) if args.marker_size_mm is not None else None,
        "mm_per_pixel_marker": mm_per_pixel_marker,

        "real_length_mm": float(args.real_length_mm) if args.real_length_mm is not None else None,
        "real_width_mm": float(args.real_width_mm) if args.real_width_mm is not None else None,

        "mouse_length_px": float(args.real_length_mm / mm_per_pixel_marker) if (args.real_length_mm and mm_per_pixel_marker) else float(aruco["side_px"] * 4.0),
        "mouse_width_px": float(args.real_width_mm / mm_per_pixel_marker) if (args.real_width_mm and mm_per_pixel_marker) else float(aruco["side_px"] * 2.0),
        "mm_per_pixel_length": mm_per_pixel_marker,
        "mm_per_pixel_width": mm_per_pixel_marker,

        "aruco_corners": [[float(x), float(y)] for x, y in aruco["corners"]],
        "note": "Single ArUco pilot ref. Origin is marker center, not physical mouse center unless offset calibration is added."
    }

    base = os.path.splitext(os.path.basename(args.image))[0]
    vis = draw_vis(image, aruco, pose)

    vis_path = os.path.join(args.output_dir, f"{base}_aruco_ref_vis.jpg")
    pose_path = os.path.join(args.output_dir, args.output_name)

    cv2.imwrite(vis_path, vis)
    save_json(pose_path, pose)

    print("=== ArUco Mouse Ref Created ===")
    print(json.dumps(pose, indent=2, ensure_ascii=False))
    print("\nSaved:")
    print(vis_path)
    print(pose_path)


if __name__ == "__main__":
    main()
