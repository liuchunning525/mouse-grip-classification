import os
import json
import argparse
from itertools import combinations

import cv2
import numpy as np


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def contour_to_list(contour):
    pts = contour.reshape(-1, 2)
    return [[int(p[0]), int(p[1])] for p in pts]


def build_full_image_roi(image):
    h, w = image.shape[:2]
    return {
        "x1": 0,
        "y1": 0,
        "x2": int(w),
        "y2": int(h),
        "width": int(w),
        "height": int(h),
    }


def detect_pink_markers(image_bgr, roi):
    x1, y1, x2, y2 = roi["x1"], roi["y1"], roi["x2"], roi["y2"]
    crop = image_bgr[y1:y2, x1:x2].copy()

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

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
        if area < 15 or area > 3000:
            continue

        x, y, w, h = cv2.boundingRect(c)
        aspect = w / h if h > 0 else 999
        if aspect < 0.25 or aspect > 4.5:
            continue

        M = cv2.moments(c)
        if abs(M["m00"]) < 1e-8:
            continue

        cx = int(M["m10"] / M["m00"]) + x1
        cy = int(M["m01"] / M["m00"]) + y1
        centers.append((cx, cy))

    return centers, mask


def choose_best_three_markers(marker_centers):
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

    return [tuple(map(int, p)) for p in best]


def order_three_markers(marker_centers):
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


def detect_mouse_contour_full_image(image, roi):
    x1, y1, x2, y2 = roi["x1"], roi["y1"], roi["x2"], roi["y2"]
    crop = image[y1:y2, x1:x2].copy()

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)

    # Dark mouse on bright desk
    _, mask_dark = cv2.threshold(blur, 120, 255, cv2.THRESH_BINARY_INV)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask_dark, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        raise RuntimeError("No contour found for mouse.")

    h, w = crop.shape[:2]
    img_area = h * w

    best = None
    best_score = -1e18

    for c in contours:
        area = cv2.contourArea(c)
        if area < img_area * 0.005 or area > img_area * 0.5:
            continue

        x, y, cw, ch = cv2.boundingRect(c)
        aspect = ch / float(cw) if cw > 0 else 0.0
        if aspect < 1.0 or aspect > 3.5:
            continue

        extent = area / float(max(cw * ch, 1))
        # prefer tall, fairly solid, central dark object
        cx = x + cw / 2.0
        cy = y + ch / 2.0
        center_penalty = abs(cx - w / 2.0) * 0.002 + abs(cy - h / 2.0) * 0.001
        score = area * 0.001 + extent * 100 + aspect * 10 - center_penalty
        if score > best_score:
            best_score = score
            best = c

    if best is None:
        raise RuntimeError("Failed to find plausible mouse contour.")

    contour = best + np.array([[[x1, y1]]], dtype=np.int32)
    return contour, mask


def contour_features(contour, real_length_mm=None, real_width_mm=None):
    pts = contour.reshape(-1, 2).astype(np.float32)

    rect = cv2.minAreaRect(pts)
    (cx, cy), (w, h), angle = rect

    if h >= w:
        mouse_length_px = float(h)
        mouse_width_px = float(w)
        theta = np.deg2rad(angle + 90.0)
    else:
        mouse_length_px = float(w)
        mouse_width_px = float(h)
        theta = np.deg2rad(angle)

    major = np.array([np.cos(theta), np.sin(theta)], dtype=np.float32)
    major = major / max(np.linalg.norm(major), 1e-8)
    minor = np.array([-major[1], major[0]], dtype=np.float32)

    mm_per_pixel_length = None
    mm_per_pixel_width = None
    if real_length_mm is not None and mouse_length_px > 1e-8:
        mm_per_pixel_length = float(real_length_mm / mouse_length_px)
    if real_width_mm is not None and mouse_width_px > 1e-8:
        mm_per_pixel_width = float(real_width_mm / mouse_width_px)

    return {
        "center_x": float(cx),
        "center_y": float(cy),
        "major_axis_x": float(major[0]),
        "major_axis_y": float(major[1]),
        "minor_axis_x": float(minor[0]),
        "minor_axis_y": float(minor[1]),
        "mouse_length_px": float(mouse_length_px),
        "mouse_width_px": float(mouse_width_px),
        "real_length_mm": float(real_length_mm) if real_length_mm is not None else None,
        "real_width_mm": float(real_width_mm) if real_width_mm is not None else None,
        "mm_per_pixel_length": mm_per_pixel_length,
        "mm_per_pixel_width": mm_per_pixel_width,
    }


def draw_vis(image, contour, features, marker_centers):
    vis = image.copy()

    cv2.drawContours(vis, [contour], -1, (0, 255, 0), 3)

    cx = int(round(features["center_x"]))
    cy = int(round(features["center_y"]))
    cv2.circle(vis, (cx, cy), 5, (0, 0, 255), -1)

    major = np.array([features["major_axis_x"], features["major_axis_y"]], dtype=np.float32)
    minor = np.array([features["minor_axis_x"], features["minor_axis_y"]], dtype=np.float32)

    half_L = features["mouse_length_px"] * 0.5
    half_W = features["mouse_width_px"] * 0.5

    p1 = (int(round(cx - major[0] * half_L)), int(round(cy - major[1] * half_L)))
    p2 = (int(round(cx + major[0] * half_L)), int(round(cy + major[1] * half_L)))
    q1 = (int(round(cx - minor[0] * half_W)), int(round(cy - minor[1] * half_W)))
    q2 = (int(round(cx + minor[0] * half_W)), int(round(cy + minor[1] * half_W)))

    cv2.line(vis, p1, p2, (255, 0, 255), 2)
    cv2.line(vis, q1, q2, (0, 165, 255), 2)

    marker_names = ["L", "M", "R"]
    for i, (x, y) in enumerate(marker_centers):
        cv2.circle(vis, (x, y), 8, (0, 255, 255), -1)
        cv2.putText(vis, marker_names[i], (x + 8, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    label = f"Center: ({cx}, {cy})"
    label2 = f"L={features['mouse_length_px']:.1f}px W={features['mouse_width_px']:.1f}px"
    cv2.putText(vis, label, (max(10, cx - 150), max(25, cy - 30)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(vis, label2, (max(10, cx - 150), max(55, cy)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    return vis


def main():
    parser = argparse.ArgumentParser(description="Standalone lock_mouse_pose without work area dependency.")
    parser.add_argument("--image", required=True, help="Mouse-only reference image")
    parser.add_argument("--output_dir", default="outputs", help="Output directory")
    parser.add_argument("--real_length_mm", type=float, default=None, help="Real mouse length in mm")
    parser.add_argument("--real_width_mm", type=float, default=None, help="Real mouse width in mm")
    parser.add_argument("--output_name", default="mouse_pose_ref.json", help="Output pose filename")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {args.image}")

    roi = build_full_image_roi(image)
    contour, mask = detect_mouse_contour_full_image(image, roi)
    features = contour_features(
        contour,
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

        "roi": roi,

        "reference_contour": contour_to_list(contour),
        "reference_markers": [[int(x), int(y)] for x, y in marker_centers],
        "reference_marker_geometry": marker_geometry
    }

    base = os.path.splitext(os.path.basename(args.image))[0]

    vis = draw_vis(image, contour, features, marker_centers)

    vis_path = os.path.join(args.output_dir, f"{base}_mouse_pose_ref_vis.jpg")
    mask_path = os.path.join(args.output_dir, f"{base}_mouse_pose_ref_mask.png")
    marker_mask_path = os.path.join(args.output_dir, f"{base}_mouse_pose_ref_marker_mask.png")
    pose_path = os.path.join(args.output_dir, args.output_name)

    cv2.imwrite(vis_path, vis)
    cv2.imwrite(mask_path, mask)
    cv2.imwrite(marker_mask_path, marker_mask)
    save_json(pose_path, pose)

    print("=== Locked Mouse Pose (standalone full-image version) ===")
    print(json.dumps(pose, indent=2, ensure_ascii=False))

    print("\nSaved:")
    print(vis_path)
    print(mask_path)
    print(marker_mask_path)
    print(pose_path)


if __name__ == "__main__":
    main()
