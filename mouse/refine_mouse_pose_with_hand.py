import os
import json
import math
import argparse
import cv2
import numpy as np
from itertools import combinations


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def normalize_vector(x, y):
    n = math.sqrt(x * x + y * y)
    if n < 1e-8:
        return 0.0, -1.0
    return x / n, y / n


def points_to_contour(points):
    return np.round(points).astype(np.int32).reshape(-1, 1, 2)


def contour_list_to_array(contour_list):
    return np.array(contour_list, dtype=np.float32)


def triangle_area(pts):
    """计算三角形面积（用于检测点是否共线）"""
    if len(pts) < 3:
        return 0
    p1, p2, p3 = pts[:3]
    return 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - 
                     (p3[0] - p1[0]) * (p2[1] - p1[1]))


def solve_rigid_transform_3points(ref_pts, cur_pts, allow_scale=False):
    """
    用 3 个点求解刚体变换（Procrustes 分析）
    
    Args:
        ref_pts: (3, 2) 参考点坐标
        cur_pts: (3, 2) 当前点坐标
        allow_scale: 是否允许缩放（默认 False，只求旋转+平移）
    
    Returns:
        R: (2, 2) 旋转矩阵
        t: (2,) 平移向量
        scale: 缩放因子
        rms_error: 拟合误差
    """
    ref_pts = np.array(ref_pts, dtype=np.float32)
    cur_pts = np.array(cur_pts, dtype=np.float32)
    
    # 计算质心
    ref_center = np.mean(ref_pts, axis=0)
    cur_center = np.mean(cur_pts, axis=0)
    
    # 去中心化
    ref_centered = ref_pts - ref_center
    cur_centered = cur_pts - cur_center
    
    if allow_scale:
        # 计算缩放因子
        ref_norm = np.linalg.norm(ref_centered)
        cur_norm = np.linalg.norm(cur_centered)
        if ref_norm < 1e-8:
            scale = 1.0
        else:
            scale = cur_norm / ref_norm
        # 缩放参考点
        ref_centered_scaled = ref_centered * scale
    else:
        scale = 1.0
        ref_centered_scaled = ref_centered
    
    # 构建协方差矩阵 H = ref_centered_scaled.T @ cur_centered
    H = ref_centered_scaled.T @ cur_centered
    
    # SVD 分解求旋转
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    # 处理反射（det 应为 1）
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # 计算平移
    t = cur_center - scale * (R @ ref_center)
    
    # 计算拟合误差
    transformed_ref = scale * (ref_pts @ R.T) + t
    rms_error = np.sqrt(np.mean(np.sum((transformed_ref - cur_pts) ** 2, axis=1)))
    
    return R, t, scale, rms_error


def apply_rigid_transform_v2(points, R, t, scale=1.0):
    """
    应用刚体变换到点集
    
    Args:
        points: (N, 2) 点集
        R: (2, 2) 旋转矩阵
        t: (2,) 平移向量
        scale: 缩放因子
    """
    points = np.array(points, dtype=np.float32)
    transformed = scale * (points @ R.T) + t
    return transformed


# =========================
# Hand mask (保持不变)
# =========================

def build_hand_mask(image_shape, hand_json, dilate_kernel=11, dilate_iter=1):
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    if hand_json.get("num_hands", 0) == 0:
        return mask

    for hand in hand_json.get("hands", []):
        pts = []
        for lm in hand.get("landmarks", []):
            pts.append([int(lm["x_px"]), int(lm["y_px"])])

        if len(pts) < 3:
            continue

        pts = np.array(pts, dtype=np.int32)
        hull = cv2.convexHull(pts)
        cv2.fillConvexPoly(mask, hull, 255)

    if dilate_kernel > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_kernel, dilate_kernel))
        mask = cv2.dilate(mask, kernel, iterations=dilate_iter)

    return mask


# =========================
# Work area
# =========================

def get_work_area_roi(work_area_json):
    roi = work_area_json["roi"]
    return {
        "x1": int(roi["x1"]),
        "y1": int(roi["y1"]),
        "x2": int(roi["x2"]),
        "y2": int(roi["y2"]),
        "width": int(roi["width"]),
        "height": int(roi["height"]),
    }


# =========================
# Pink marker detection (改进版)
# =========================

def detect_pink_markers_v2(image_bgr, work_area_roi=None, debug=False):
    """改进版 marker 检测，带参数调优"""
    if work_area_roi is not None:
        x1, y1, x2, y2 = work_area_roi["x1"], work_area_roi["y1"], work_area_roi["x2"], work_area_roi["y2"]
        roi = image_bgr[y1:y2, x1:x2].copy()
    else:
        x1, y1 = 0, 0
        roi = image_bgr.copy()

    # 高斯模糊降噪
    roi = cv2.GaussianBlur(roi, (3, 3), 0)
    
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # 收紧的粉色范围
    lower = np.array([145, 80, 80], dtype=np.uint8)
    upper = np.array([175, 255, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower, upper)

    # 形态学操作
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    centers = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 20 or area > 1500:  # 调整面积范围
            continue

        # 检查形状（圆形度）
        perimeter = cv2.arcLength(c, True)
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity < 0.5:  # 太不圆就跳过
                continue

        M = cv2.moments(c)
        if abs(M["m00"]) < 1e-8:
            continue

        cx = int(M["m10"] / M["m00"]) + x1
        cy = int(M["m01"] / M["m00"]) + y1
        centers.append((cx, cy))

    if debug:
        return centers, mask, roi
    return centers, mask


def select_best_three_markers_v2(marker_centers):
    """
    选择最优的 3 个 marker（基于几何约束）
    优先选择三角形面积大、共线性小的组合
    """
    if len(marker_centers) < 3:
        return None
    if len(marker_centers) == 3:
        return sorted(marker_centers, key=lambda p: p[0])

    pts = np.array(marker_centers, dtype=np.float32)
    best = None
    best_score = -1e18

    for idxs in combinations(range(len(pts)), 3):
        sub = pts[list(idxs)]
        
        # 计算三角形面积（越大越好）
        area = triangle_area(sub)
        
        # 计算最小边长
        dists = []
        for i in range(3):
            for j in range(i+1, 3):
                d = np.linalg.norm(sub[i] - sub[j])
                dists.append(d)
        min_dist = min(dists)
        
        # 综合评分：面积大 + 最小边长大
        score = area + 0.5 * min_dist
        
        if score > best_score:
            best_score = score
            best = sub

    best = [tuple(map(int, p)) for p in best]
    best = sorted(best, key=lambda p: p[0])
    return best


# =========================
# 诊断函数
# =========================

def diagnose_markers(ref_markers, cur_markers):
    """诊断 marker 匹配误差"""
    print("\n=== Marker 诊断 ===")
    
    ref_arr = np.array(ref_markers)
    cur_arr = np.array(cur_markers)
    
    for i, (ref, cur) in enumerate(zip(ref_arr, cur_arr)):
        dx = cur[0] - ref[0]
        dy = cur[1] - ref[1]
        dist = np.sqrt(dx**2 + dy**2)
        print(f"  Marker {i+1}: 偏移 ({dx:.1f}, {dy:.1f}) 距离 {dist:.1f}px")
    
    # 检查三角形面积变化
    ref_area = triangle_area(ref_arr)
    cur_area = triangle_area(cur_arr)
    area_ratio = cur_area / ref_area if ref_area > 0 else 1.0
    print(f"  三角形面积: ref={ref_area:.1f}, cur={cur_area:.1f}, 比例={area_ratio:.3f}")
    
    # 检查形状一致性（边长比）
    print("  边长比例:")
    for i in range(3):
        for j in range(i+1, 3):
            ref_dist = np.linalg.norm(ref_arr[i] - ref_arr[j])
            cur_dist = np.linalg.norm(cur_arr[i] - cur_arr[j])
            ratio = cur_dist / ref_dist if ref_dist > 0 else 1.0
            print(f"    {i+1}-{j+1}: ref={ref_dist:.1f}, cur={cur_dist:.1f}, 比例={ratio:.3f}")
    
    return ref_area, cur_area


# =========================
# 主函数：改进版位姿估计
# =========================

def refine_mouse_pose_v2(ref_image_bgr, cur_image_bgr, ref_pose, work_area_roi, hand_json, allow_scale=False):
    """
    改进版：使用 3 点刚体变换估计鼠标位姿
    """
    vis = cur_image_bgr.copy()
    
    # 构建手部掩码
    hand_mask_full = build_hand_mask(cur_image_bgr.shape, hand_json, dilate_kernel=11, dilate_iter=1)
    
    # 检查参考点
    if "reference_markers" not in ref_pose or len(ref_pose["reference_markers"]) < 3:
        raise RuntimeError("reference_markers missing in mouse_pose_ref.json")
    
    ref_markers = [tuple(p) for p in ref_pose["reference_markers"]]
    
    # 检测当前图像的 marker
    cur_markers, cur_mask = detect_pink_markers_v2(cur_image_bgr, work_area_roi)
    cur_markers = select_best_three_markers_v2(cur_markers)
    
    if cur_markers is None or len(cur_markers) < 3:
        raise RuntimeError("Failed to detect 3 pink markers in current image.")
    
    # 按 x 坐标排序
    ref_markers = sorted(ref_markers, key=lambda p: p[0])
    cur_markers = sorted(cur_markers, key=lambda p: p[0])
    
    # 诊断
    ref_area, cur_area = diagnose_markers(ref_markers, cur_markers)
    
    # 求解 3 点刚体变换
    R, t, scale, rms_error = solve_rigid_transform_3points(
        ref_markers, cur_markers, allow_scale=allow_scale
    )
    
    print(f"\n=== 刚体变换结果 ===")
    print(f"  旋转矩阵 R: {R}")
    print(f"  平移向量 t: {t}")
    print(f"  缩放因子 scale: {scale:.4f}")
    print(f"  拟合误差 RMS: {rms_error:.2f}px")
    
    # 应用变换到鼠标中心
    ref_center = np.array([[ref_pose["center_x"], ref_pose["center_y"]]], dtype=np.float32)
    cur_center = apply_rigid_transform_v2(ref_center, R, t, scale)[0]
    
    # 应用变换到鼠标主轴方向
    ref_major = np.array([ref_pose["major_axis_x"], ref_pose["major_axis_y"]], dtype=np.float32)
    # 只旋转，不平移
    cur_major = ref_major @ R.T
    cur_major = normalize_vector(cur_major[0], cur_major[1])
    cur_major = np.array(cur_major, dtype=np.float32)
    cur_minor = np.array([-cur_major[1], cur_major[0]], dtype=np.float32)
    
    # 计算旋转角度差
    ref_angle = math.atan2(ref_pose["major_axis_y"], ref_pose["major_axis_x"])
    cur_angle = math.atan2(cur_major[1], cur_major[0])
    angle_delta_deg = math.degrees(cur_angle - ref_angle)
    
    # 构建当前位姿
    current_pose = {
        "center_x": float(cur_center[0]),
        "center_y": float(cur_center[1]),
        "major_axis_x": float(cur_major[0]),
        "major_axis_y": float(cur_major[1]),
        "minor_axis_x": float(cur_minor[0]),
        "minor_axis_y": float(cur_minor[1]),
        "mouse_length_px": float(ref_pose["mouse_length_px"]),
        "mouse_width_px": float(ref_pose["mouse_width_px"]),
        "real_length_mm": ref_pose.get("real_length_mm"),
        "real_width_mm": ref_pose.get("real_width_mm"),
        "mm_per_pixel_length": ref_pose.get("mm_per_pixel_length"),
        "mm_per_pixel_width": ref_pose.get("mm_per_pixel_width"),
        
        # 变换参数
        "dx_from_ref": float(cur_center[0] - ref_pose["center_x"]),
        "dy_from_ref": float(cur_center[1] - ref_pose["center_y"]),
        "angle_delta_deg": float(angle_delta_deg),
        "scale": float(scale),
        "rms_error_px": float(rms_error),
        
        # marker 信息
        "reference_markers": [[float(x), float(y)] for x, y in ref_markers],
        "current_markers": [[float(x), float(y)] for x, y in cur_markers],
        "triangle_area_ratio": float(cur_area / ref_area) if ref_area > 0 else 1.0,
    }
    
    # 变换鼠标轮廓
    transformed_contour = None
    if "reference_contour" in ref_pose and ref_pose["reference_contour"]:
        ref_contour = contour_list_to_array(ref_pose["reference_contour"])
        transformed_contour = apply_rigid_transform_v2(ref_contour, R, t, scale)
        current_pose["transformed_contour"] = [[float(p[0]), float(p[1])] for p in transformed_contour]
    
    # =========================
    # 可视化
    # =========================
    
    # 绘制工作区域
    x1, y1, x2, y2 = work_area_roi["x1"], work_area_roi["y1"], work_area_roi["x2"], work_area_roi["y2"]
    cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 255, 0), 2)
    
    # 绘制手部掩码（半透明）
    overlay = vis.copy()
    overlay[hand_mask_full > 0] = (0, 0, 180)
    vis = cv2.addWeighted(vis, 0.85, overlay, 0.15, 0)
    
    # 绘制参考位姿（黄色）
    def draw_pose(img, pose, color_center, color_major, color_minor, label, offset_x=0, offset_y=0):
        cx = pose["center_x"] + offset_x
        cy = pose["center_y"] + offset_y
        ux = pose["major_axis_x"]
        uy = pose["major_axis_y"]
        vx = pose["minor_axis_x"]
        vy = pose["minor_axis_y"]
        L = pose["mouse_length_px"]
        W = pose["mouse_width_px"]
        
        half_L = int(round(L * 0.5))
        half_W = int(round(W * 0.5))
        
        center = (int(round(cx)), int(round(cy)))
        major_p1 = (int(round(cx - ux * half_L)), int(round(cy - uy * half_L)))
        major_p2 = (int(round(cx + ux * half_L)), int(round(cy + uy * half_L)))
        minor_p1 = (int(round(cx - vx * half_W)), int(round(cy - vy * half_W)))
        minor_p2 = (int(round(cx + vx * half_W)), int(round(cy + vy * half_W)))
        
        cv2.circle(img, center, 5, color_center, -1)
        cv2.line(img, major_p1, major_p2, color_major, 2)
        cv2.line(img, minor_p1, minor_p2, color_minor, 2)
        cv2.putText(img, label, (center[0] + 10, center[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_center, 2)
    
    # 绘制参考位姿（偏移显示，避免重叠）
    draw_pose(vis, ref_pose,
              color_center=(0, 255, 255),
              color_major=(0, 255, 255),
              color_minor=(0, 200, 200),
              label="ref")
    
    # 绘制当前位姿（红色）
    draw_pose(vis, current_pose,
              color_center=(0, 0, 255),
              color_major=(255, 0, 255),
              color_minor=(0, 165, 255),
              label="current")
    
    # 绘制 marker
    for i, (x, y) in enumerate(cur_markers):
        cv2.circle(vis, (int(x), int(y)), 8, (0, 255, 255), -1)
        cv2.putText(vis, f"C{i+1}", (int(x) + 8, int(y) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    
    # 绘制变换后的轮廓
    if transformed_contour is not None:
        contour_pts = points_to_contour(transformed_contour)
        cv2.drawContours(vis, [contour_pts], -1, (0, 255, 0), 3)
    
    # 绘制信息文本
    y_offset = 35
    cv2.putText(vis, f"dx={current_pose['dx_from_ref']:.1f}, dy={current_pose['dy_from_ref']:.1f}",
                (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(vis, f"angle={current_pose['angle_delta_deg']:.1f} deg",
                (20, y_offset + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(vis, f"RMS error={rms_error:.1f}px, scale={scale:.3f}",
                (20, y_offset + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # 如果误差过大，显示警告
    if rms_error > 15:
        cv2.putText(vis, "WARNING: High RMS error! Check markers.",
                    (20, y_offset + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    debug = {
        "hand_mask_full": hand_mask_full,
        "cur_marker_mask": cur_mask,
        "R": R.tolist(),
        "t": t.tolist(),
        "rms_error": rms_error
    }
    
    return current_pose, vis, debug


# =========================
# Main
# =========================

def main():
    parser = argparse.ArgumentParser(description="3-point rigid transform for mouse pose estimation")
    parser.add_argument("--ref_image", required=True, help="Mouse-only reference image")
    parser.add_argument("--cur_image", required=True, help="Current hand-on-mouse image")
    parser.add_argument("--ref_pose", required=True, help="Reference mouse pose json")
    parser.add_argument("--work_area_json", required=True, help="Work area json")
    parser.add_argument("--hand_json", required=True, help="Current hand landmark json")
    parser.add_argument("--output_dir", default="outputs", help="Output directory")
    parser.add_argument("--output_name", default="mouse_pose_current.json", help="Output pose filename")
    parser.add_argument("--allow_scale", action="store_true", help="Allow scaling in transform")
    args = parser.parse_args()

    ensure_dir(args.output_dir)

    ref_image = cv2.imread(args.ref_image)
    cur_image = cv2.imread(args.cur_image)

    if ref_image is None:
        raise FileNotFoundError(f"Reference image not found: {args.ref_image}")
    if cur_image is None:
        raise FileNotFoundError(f"Current image not found: {args.cur_image}")

    ref_pose = load_json(args.ref_pose)
    work_area = load_json(args.work_area_json)
    hand_json = load_json(args.hand_json)

    work_area_roi = get_work_area_roi(work_area)

    current_pose, vis, debug = refine_mouse_pose_v2(
        ref_image_bgr=ref_image,
        cur_image_bgr=cur_image,
        ref_pose=ref_pose,
        work_area_roi=work_area_roi,
        hand_json=hand_json,
        allow_scale=args.allow_scale
    )

    base = os.path.splitext(os.path.basename(args.cur_image))[0]

    pose_path = os.path.join(args.output_dir, args.output_name)
    vis_path = os.path.join(args.output_dir, f"{base}_mouse_pose_refined_v2.jpg")
    hand_mask_path = os.path.join(args.output_dir, f"{base}_hand_mask_v2.png")
    marker_mask_path = os.path.join(args.output_dir, f"{base}_marker_mask_v2.png")

    save_json(pose_path, current_pose)
    cv2.imwrite(vis_path, vis)
    cv2.imwrite(hand_mask_path, debug["hand_mask_full"])
    cv2.imwrite(marker_mask_path, debug["cur_marker_mask"])

    print("\n=== Refined Mouse Pose (3-point rigid transform) ===")
    print(json.dumps(current_pose, indent=2, ensure_ascii=False))

    print("\nSaved:")
    print(f"  Pose: {pose_path}")
    print(f"  Visualization: {vis_path}")
    print(f"  Hand mask: {hand_mask_path}")
    print(f"  Marker mask: {marker_mask_path}")
    
    if debug["rms_error"] > 15:
        print("\n⚠️  WARNING: RMS error > 15px. Please check:")
        print("  1. Markers are clearly visible (not covered by hand)")
        print("  2. Lighting is consistent")
        print("  3. Markers are attached firmly to the mouse")


if __name__ == "__main__":
    main()