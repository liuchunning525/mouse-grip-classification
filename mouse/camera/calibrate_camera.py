import cv2
import numpy as np
import glob
import os


# ====== 你需要改的参数 ======
image_folder = "calibration_images"
image_extension = "*.jpg"   # 如果你是 png，就改成 *.png
chessboard_size = (6, 8)    # 内角点数量，不是格子数
square_size = 0.02         # 每个小格子的实际边长，单位：米
# ===========================


def main():
    # 终止条件：亚像素角点优化
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001
    )

    # 构造棋盘格在真实世界中的 3D 点
    # 例如 (0,0,0), (1,0,0), (2,0,0) ... 然后乘 square_size
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    objp *= square_size

    # 存储所有图像中的 3D 点和 2D 点
    objpoints = []  # 世界坐标中的点
    imgpoints = []  # 图像坐标中的点

    image_paths = glob.glob(os.path.join(image_folder, image_extension))

    if not image_paths:
        print("没有找到图片，请检查文件夹路径和扩展名。")
        return

    print(f"共找到 {len(image_paths)} 张图片")

    valid_count = 0
    image_size = None

    # 保存识别结果图
    output_folder = "calibration_detected"
    os.makedirs(output_folder, exist_ok=True)

    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            print(f"无法读取图片: {path}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        image_size = gray.shape[::-1]

        # 查找棋盘格角点
        found, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

        if found:
            valid_count += 1
            objpoints.append(objp)

            # 亚像素优化
            corners2 = cv2.cornerSubPix(
                gray,
                corners,
                (11, 11),
                (-1, -1),
                criteria
            )
            imgpoints.append(corners2)

            # 画出角点，方便检查
            drawn = img.copy()
            cv2.drawChessboardCorners(drawn, chessboard_size, corners2, found)

            save_path = os.path.join(output_folder, os.path.basename(path))
            cv2.imwrite(save_path, drawn)

            print(f"[成功] 检测到角点: {path}")
        else:
            print(f"[失败] 没检测到角点: {path}")

    print(f"\n成功用于标定的图片数量: {valid_count}")

    if valid_count < 10:
        print("可用图片太少，建议至少 10~15 张成功识别的图片。")
        return

    # 相机标定
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints,
        imgpoints,
        image_size,
        None,
        None
    )

    print("\n===== 标定结果 =====")
    print("重投影误差 ret =", ret)
    print("\n相机矩阵 camera_matrix =")
    print(camera_matrix)
    print("\n畸变系数 dist_coeffs =")
    print(dist_coeffs)

    # 计算平均重投影误差
    total_error = 0
    for i in range(len(objpoints)):
        projected_points, _ = cv2.projectPoints(
            objpoints[i],
            rvecs[i],
            tvecs[i],
            camera_matrix,
            dist_coeffs
        )
        error = cv2.norm(imgpoints[i], projected_points, cv2.NORM_L2) / len(projected_points)
        total_error += error

    mean_error = total_error / len(objpoints)
    print("\n平均重投影误差 =", mean_error)

    # 保存参数
    np.savez(
        "camera_calibration.npz",
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        rvecs=rvecs,
        tvecs=tvecs
    )
    print("\n参数已保存到 camera_calibration.npz")

    # 测试去畸变
    test_img = cv2.imread(image_paths[0])
    undistorted = cv2.undistort(test_img, camera_matrix, dist_coeffs)
    cv2.imwrite("undistorted_example.jpg", undistorted)
    print("已保存去畸变示例图: undistorted_example.jpg")


if __name__ == "__main__":
    main()