import cv2
import numpy as np

img_pts = np.array([
    [166, 57],   # 左上
    [111, 459],  # 左下
    [681, 50],   # 右上
    [662, 448]   # 右下
], dtype=np.float32)

world_pts = np.array([
    [0, 5],   # 左上
    [0, 0],   # 左下
    [6, 5],   # 右上
    [6, 0]    # 右下
], dtype=np.float32)

H, _ = cv2.findHomography(img_pts, world_pts)

print("Homography matrix:")
print(H)

np.save("H_matrix.npy", H)