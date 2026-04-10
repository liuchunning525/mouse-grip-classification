
import os
import json
import argparse
import cv2
import numpy as np

from detect_mouse_in_work_area import detect_mouse_in_roi, load_json


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def contour_to_list(contour):
    pts = contour.reshape(-1, 2)
    return [[int(p[0]), int(p[1])] for p in pts]


def detect_pink_markers(image_bgr, roi):
    x1, y1, x2, y2 = roi["x1"], roi["y1"], roi["x2"], roi["y2"]
    crop = image_bgr[y1:y2, x1:x2].copy()

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    # 粉色贴纸阈值
    lower = np.array([140, 60, 60], dtype=np.uint8)
    upper = np.array([179, 255, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower, upper)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    centers = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 15 or area > 2000:
            continue

        x, y, w, h = cv2.boundingRect(c)
        aspect = w / h if h > 0 else 999
        if aspect < 0.3 or aspect > 3.0:
            continue

        M = cv2.moments(c)
        if abs(M["m00"]) < 1e-8:
            continue

        cx = int(M["m10"] / M["m00"]) + x1
        cy = int(M["m01"] / M["m00"]) + y1
        centers.append((cx, cy))

    return centers, mask


def choose_best_three_markers(marker_centers):
    """
    从候选点中选择最可能是三个底部贴纸的组合：
    - 三点尽量共线（y_std 小）
    - 左右跨度尽量大（x_span 大）
    """
    if len(marker_centers) < 3:
        return None
    if len(marker_centers) == 3:
        return list(marker_centers)

    pts = np.array(marker_centers, dtype=np.float32)
    best = None
    best_score = 1e18

    from itertools import combinations
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

    return [tuple(map(int, p)) for p in best]


def order_three_markers(marker_centers):
    """
    将三个 marker 稳定排序为 [左外点, 中点, 右外点]
    做法：
    1. 中点 = 到另外两点距离和最小的点
    2. 其余两点作为外点，再按 x 排成左右
    """
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
        (int(round(left_pt[0])), int(round(left_pt[1]))),
        (int(round(middle_pt[0])), int(round(middle_pt[1]))),
        (int(round(right_pt[0])), int(round(right_pt[1])))
    ]
    return ordered


def compute_marker_geometry(markers):
    pts = np.array(markers, dtype=np.float32)
    left, middle, right = pts

    return {
        "outer_distance": float(np.linalg.norm(right - left)),
        "left_middle_distance": float(np.linalg.norm(middle - left)),
        "middle_right_distance": float(np.linalg.norm(right - middle)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Mouse-only reference image")
    parser.add_argument("--work_area_json", required=True, help="Work area json from detect_work_area.py")
    parser.add_argument("--output_dir", default="outputs", help="Output directory")
    parser.add_argument("--real_length_mm", type=float, default=None, help="Real mouse length in mm")
    parser.add_argument("--real_width_mm", type=float, default=None, help="Real mouse width in mm")
    parser.add_argument("--output_name", default="mouse_pose_ref.json", help="Output pose filename")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {args.image}")

    work_area = load_json(args.work_area_json)
    roi = work_area["roi"]

    features, vis, mask, contour = detect_mouse_in_roi(
        image=image,
        roi=roi,
        real_length_mm=args.real_length_mm,
        real_width_mm=args.real_width_mm
    )

    marker_centers, marker_mask = detect_pink_markers(image, roi)
    marker_centers = choose_best_three_markers(marker_centers)
    marker_centers = order_three_markers(marker_centers)

    if marker_centers is None or len(marker_centers) < 3:
        raise RuntimeError("Failed to detect and order 3 pink markers in reference image.")

    marker_geometry = compute_marker_geometry(marker_centers)

    pose = {
        "source_image": os.path.basename(args.image),
        "work_area_json": os.path.basename(args.work_area_json),

        "center_x": features["center_x"],
        "center_y": features["center_y"],

        "major_axis_x": features["major_axis_x"],
        "major_axis_y": features["major_axis_y"],
        "minor_axis_x": features["minor_axis_x"],
        "minor_axis_y": features["minor_axis_y"],

        "mouse_length_px": features["mouse_length_px"],
        "mouse_width_px": features["mouse_width_px"],

        "real_length_mm": features["real_length_mm"],
        "real_width_mm": features["real_width_mm"],
        "mm_per_pixel_length": features["mm_per_pixel_length"],
        "mm_per_pixel_width": features["mm_per_pixel_width"],

        "roi": features["roi"],

        "reference_contour": contour_to_list(contour),
        "reference_markers": [[int(x), int(y)] for x, y in marker_centers],
        "reference_marker_geometry": marker_geometry
    }

    base = os.path.splitext(os.path.basename(args.image))[0]

    vis_markers = vis.copy()
    marker_names = ["L", "M", "R"]
    for i, (x, y) in enumerate(marker_centers):
        cv2.circle(vis_markers, (x, y), 8, (0, 255, 255), -1)
        cv2.putText(vis_markers, marker_names[i], (x + 8, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    vis_path = os.path.join(args.output_dir, f"{base}_mouse_pose_ref_vis.jpg")
    mask_path = os.path.join(args.output_dir, f"{base}_mouse_pose_ref_mask.png")
    marker_mask_path = os.path.join(args.output_dir, f"{base}_mouse_pose_ref_marker_mask.png")
    pose_path = os.path.join(args.output_dir, args.output_name)

    cv2.imwrite(vis_path, vis_markers)
    cv2.imwrite(mask_path, mask)
    cv2.imwrite(marker_mask_path, marker_mask)
    save_json(pose_path, pose)

    print("=== Locked Mouse Pose (stable 2-outer-point version) ===")
    print(json.dumps(pose, indent=2, ensure_ascii=False))

    print("\nSaved:")
    print(vis_path)
    print(mask_path)
    print(marker_mask_path)
    print(pose_path)


if __name__ == "__main__":
    main()
