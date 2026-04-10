import os
import cv2
import json
import argparse
import numpy as np


def normalize_vector(x, y):
    norm = np.sqrt(x * x + y * y)
    if norm < 1e-8:
        return 0.0, -1.0
    return x / norm, y / norm


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def keep_largest_component(mask):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask.copy(), None

    largest_idx = -1
    largest_area = 0
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > largest_area:
            largest_area = area
            largest_idx = i

    out = np.zeros_like(mask)
    out[labels == largest_idx] = 255
    return out, stats[largest_idx]


def fill_component_holes(mask):
    h, w = mask.shape
    flood = mask.copy()
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 255)
    flood_inv = cv2.bitwise_not(flood)
    filled = cv2.bitwise_or(mask, flood_inv)
    return filled


def remove_corner_regions(mask, radius=28):
    h, w = mask.shape[:2]
    corners = [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]
    for cx, cy in corners:
        cv2.circle(mask, (cx, cy), radius, 0, -1)
    return mask


def preprocess_for_mouse_mask(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur_bg = cv2.GaussianBlur(gray, (0, 0), 31)
    norm = cv2.divide(gray, blur_bg, scale=255)

    _, th1 = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    th2 = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        51,
        7,
    )

    mask = cv2.bitwise_or(th1, th2)

    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    k7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    k11 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k3, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k7, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k11, iterations=1)

    mask = remove_corner_regions(mask, radius=28)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask

    h, w = mask.shape
    roi_cx = w / 2.0
    roi_cy = h / 2.0

    best_idx = -1
    best_score = -1e18

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        ww = stats[i, cv2.CC_STAT_WIDTH]
        hh = stats[i, cv2.CC_STAT_HEIGHT]

        if area < 2000:
            continue
        if ww < 60 or hh < 120:
            continue

        cx, cy = centroids[i]
        dist_center = np.sqrt((cx - roi_cx) ** 2 + (cy - roi_cy) ** 2)

        aspect = hh / max(ww, 1)
        rect_area = max(ww * hh, 1)
        extent = area / rect_area

        score = 0.0
        score += area * 0.02
        score += extent * 120.0
        score += max(0.0, 1.0 - abs(aspect - 1.8)) * 90.0
        score -= dist_center * 0.18

        if score > best_score:
            best_score = score
            best_idx = i

    if best_idx == -1:
        return mask

    out = np.zeros_like(mask)
    out[labels == best_idx] = 255
    out = fill_component_holes(out)
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, k5, iterations=2)
    return out


def contour_score(contour, roi_shape):
    area = cv2.contourArea(contour)
    if area < 2000:
        return -1e9

    x, y, w, h = cv2.boundingRect(contour)
    if w <= 0 or h <= 0:
        return -1e9

    rect_area = w * h
    extent = area / rect_area

    perimeter = cv2.arcLength(contour, True)
    if perimeter <= 0:
        return -1e9

    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 1e-8 else 0.0

    circularity = 4.0 * np.pi * area / (perimeter * perimeter)

    rr = cv2.minAreaRect(contour)
    (_, _), (rw, rh), _ = rr
    if rw < 1 or rh < 1:
        return -1e9

    long_side = max(rw, rh)
    short_side = min(rw, rh)
    aspect_ratio = long_side / short_side if short_side > 0 else 999.0

    if aspect_ratio < 1.15 or aspect_ratio > 3.3:
        return -1e9

    roi_h, roi_w = roi_shape[:2]
    cx = x + w / 2.0
    cy = y + h / 2.0
    roi_cx = roi_w / 2.0
    roi_cy = roi_h / 2.0
    dist_center = np.sqrt((cx - roi_cx) ** 2 + (cy - roi_cy) ** 2)

    score = 0.0
    score += area * 0.015
    score += extent * 120.0
    score += solidity * 140.0
    score += circularity * 30.0
    score += max(0.0, 1.0 - abs(aspect_ratio - 1.8)) * 80.0
    score -= dist_center * 0.15
    return score


def find_best_mouse_contour(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    best_contour = None
    best_score = -1e18
    for c in contours:
        s = contour_score(c, mask.shape)
        if s > best_score:
            best_score = s
            best_contour = c
    return best_contour


def compute_pca_axis(contour):
    pts = contour.reshape(-1, 2).astype(np.float32)
    mean = np.mean(pts, axis=0)

    centered = pts - mean
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eig(cov)

    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    ux, uy = float(eigvecs[0, 0]), float(eigvecs[1, 0])
    vx, vy = float(eigvecs[0, 1]), float(eigvecs[1, 1])

    ux, uy = normalize_vector(ux, uy)
    vx, vy = normalize_vector(vx, vy)

    if uy > 0:
        ux, uy = -ux, -uy
        vx, vy = -vx, -vy

    return mean, (ux, uy), (vx, vy), eigvals


def remove_bottom_cable_by_row_width(mask):
    work = mask.copy()
    h, w = work.shape

    row_widths = np.sum(work > 0, axis=1).astype(np.int32)
    if np.max(row_widths) <= 0:
        return work

    max_width = int(np.max(row_widths))
    narrow_threshold = max(12, int(max_width * 0.23))
    start_check_y = int(h * 0.72)

    cut_y = None
    narrow_count = 0

    for y in range(h - 1, start_check_y - 1, -1):
        if row_widths[y] <= narrow_threshold:
            narrow_count += 1
        else:
            if narrow_count >= 12:
                cut_y = y + 1
                break
            narrow_count = 0

    if cut_y is not None:
        work[cut_y:, :] = 0

    work, _ = keep_largest_component(work)
    work = fill_component_holes(work)
    return work


def rebuild_mask_from_outer_contour(mask):
    contour = find_best_mouse_contour(mask)
    if contour is None:
        return mask

    hull = cv2.convexHull(contour)
    rebuilt = np.zeros_like(mask)
    cv2.drawContours(rebuilt, [hull], -1, 255, -1)

    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    k9 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    rebuilt = cv2.morphologyEx(rebuilt, cv2.MORPH_OPEN, k5, iterations=1)
    rebuilt = cv2.morphologyEx(rebuilt, cv2.MORPH_CLOSE, k9, iterations=2)
    return rebuilt


def smooth_and_adjust_contour(contour):
    peri = cv2.arcLength(contour, True)
    epsilon1 = 0.0025 * peri
    contour = cv2.approxPolyDP(contour, epsilon1, True)

    hull = cv2.convexHull(contour)

    mask = np.zeros(
        (
            max(1, np.max(hull[:, 0, 1]) + 20),
            max(1, np.max(hull[:, 0, 0]) + 20),
        ),
        dtype=np.uint8,
    )
    cv2.drawContours(mask, [hull], -1, 255, -1)

    k7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k7, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k7, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return hull

    contour = max(contours, key=cv2.contourArea)
    epsilon2 = 0.0035 * cv2.arcLength(contour, True)
    contour = cv2.approxPolyDP(contour, epsilon2, True)
    return contour


def detect_mouse_in_roi(image, roi, real_length_mm=None, real_width_mm=None):
    x1, y1, x2, y2 = roi["x1"], roi["y1"], roi["x2"], roi["y2"]

    roi_img = image[y1:y2, x1:x2].copy()
    vis = image.copy()

    scale = 0.7
    small_roi = cv2.resize(roi_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    mask_small = preprocess_for_mouse_mask(small_roi)
    mask = cv2.resize(mask_small, (roi_img.shape[1], roi_img.shape[0]), interpolation=cv2.INTER_NEAREST)

    mask = remove_corner_regions(mask, radius=30)
    mask = remove_bottom_cable_by_row_width(mask)
    mask = rebuild_mask_from_outer_contour(mask)
    mask, _ = keep_largest_component(mask)
    mask = fill_component_holes(mask)

    contour_local = find_best_mouse_contour(mask)
    if contour_local is None:
        raise RuntimeError("Mouse contour not found inside work area.")

    contour_local = smooth_and_adjust_contour(contour_local)

    final_mask = np.zeros_like(mask)
    cv2.drawContours(final_mask, [contour_local], -1, 255, -1)
    final_mask = cv2.morphologyEx(
        final_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        iterations=2,
    )

    contour_local = find_best_mouse_contour(final_mask)
    if contour_local is None:
        raise RuntimeError("Final mouse contour not found.")

    contour = contour_local.copy()
    contour[:, 0, 0] += x1
    contour[:, 0, 1] += y1

    area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, True))

    bx, by, bw, bh = cv2.boundingRect(contour)

    rr = cv2.minAreaRect(contour)
    (cx, cy), (rw, rh), angle = rr
    box = cv2.boxPoints(rr).astype(np.int32)

    long_side = float(max(rw, rh))
    short_side = float(min(rw, rh))

    _, major_axis, minor_axis, _ = compute_pca_axis(contour)

    aspect_ratio = long_side / short_side if short_side > 1e-8 else None
    extent = area / float(bw * bh) if bw * bh > 0 else None
    circularity = 4.0 * np.pi * area / (perimeter * perimeter) if perimeter > 1e-8 else None

    mm_per_pixel_length = None
    mm_per_pixel_width = None

    if real_length_mm is not None and long_side > 1e-8:
        mm_per_pixel_length = float(real_length_mm) / long_side
    if real_width_mm is not None and short_side > 1e-8:
        mm_per_pixel_width = float(real_width_mm) / short_side

    cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 255, 0), 2)
    cv2.drawContours(vis, [contour], -1, (0, 255, 0), 3)
    cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), (255, 0, 0), 2)
    cv2.drawContours(vis, [box], -1, (0, 255, 255), 1)

    center = (int(round(cx)), int(round(cy)))
    cv2.circle(vis, center, 5, (0, 0, 255), -1)

    half_long = int(round(long_side * 0.5))
    half_short = int(round(short_side * 0.5))

    ux, uy = major_axis
    vx, vy = minor_axis

    major_p1 = (int(round(cx - ux * half_long)), int(round(cy - uy * half_long)))
    major_p2 = (int(round(cx + ux * half_long)), int(round(cy + uy * half_long)))
    minor_p1 = (int(round(cx - vx * half_short)), int(round(cy - vy * half_short)))
    minor_p2 = (int(round(cx + vx * half_short)), int(round(cy + vy * half_short)))

    cv2.line(vis, major_p1, major_p2, (255, 0, 255), 2)
    cv2.line(vis, minor_p1, minor_p2, (0, 165, 255), 2)

    cv2.putText(
        vis,
        f"Center: ({int(round(cx))}, {int(round(cy))})",
        (bx, max(20, by - 30)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )
    cv2.putText(
        vis,
        f"L={long_side:.1f}px W={short_side:.1f}px",
        (bx, max(45, by - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
    )

    features = {
        "roi": roi,
        "contour_area_px2": area,
        "contour_perimeter_px": perimeter,
        "bbox_x": int(bx),
        "bbox_y": int(by),
        "bbox_w": int(bw),
        "bbox_h": int(bh),
        "center_x": float(cx),
        "center_y": float(cy),
        "mouse_length_px": long_side,
        "mouse_width_px": short_side,
        "aspect_ratio": float(aspect_ratio) if aspect_ratio is not None else None,
        "extent": float(extent) if extent is not None else None,
        "circularity": float(circularity) if circularity is not None else None,
        "major_axis_x": float(major_axis[0]),
        "major_axis_y": float(major_axis[1]),
        "minor_axis_x": float(minor_axis[0]),
        "minor_axis_y": float(minor_axis[1]),
        "real_length_mm": float(real_length_mm) if real_length_mm is not None else None,
        "real_width_mm": float(real_width_mm) if real_width_mm is not None else None,
        "mm_per_pixel_length": mm_per_pixel_length,
        "mm_per_pixel_width": mm_per_pixel_width,
        "reference_contour": contour.reshape(-1, 2).astype(float).tolist(),
    }

    return features, vis, final_mask, contour


def save_outputs(features, vis, mask, output_dir, image_path):
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(image_path))[0]

    vis_path = os.path.join(output_dir, f"{base}_mouse_in_area_vis.jpg")
    mask_path = os.path.join(output_dir, f"{base}_mouse_in_area_mask.png")
    json_path = os.path.join(output_dir, f"{base}_mouse_in_area.json")

    cv2.imwrite(vis_path, vis)
    cv2.imwrite(mask_path, mask)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(features, f, indent=2, ensure_ascii=False)

    return vis_path, mask_path, json_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="输入图片")
    parser.add_argument("--work_area_json", required=True, help="detect_work_area.py 输出的工作区 json")
    parser.add_argument("--output_dir", default="outputs", help="输出目录")
    parser.add_argument("--real_length_mm", type=float, default=None, help="鼠标真实长度（毫米）")
    parser.add_argument("--real_width_mm", type=float, default=None, help="鼠标真实宽度（毫米）")
    args = parser.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {args.image}")

    work_area = load_json(args.work_area_json)
    roi = work_area["roi"]

    features, vis, mask, contour = detect_mouse_in_roi(
        image=image,
        roi=roi,
        real_length_mm=args.real_length_mm,
        real_width_mm=args.real_width_mm,
    )

    vis_path, mask_path, json_path = save_outputs(features, vis, mask, args.output_dir, args.image)

    print("=== Mouse Detection In Work Area (Improved) ===")
    print(json.dumps(features, indent=2, ensure_ascii=False))
    print()
    print("Saved:")
    print(vis_path)
    print(mask_path)
    print(json_path)


if __name__ == "__main__":
    main()