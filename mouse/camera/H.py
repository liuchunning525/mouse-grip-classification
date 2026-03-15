import cv2
import numpy as np

H = np.load("H_matrix.npy")

def pixel_to_world(x, y):
    p = np.array([x, y, 1.0], dtype=np.float32)
    w = H @ p
    w = w / w[2]
    return float(w[0]), float(w[1])

img = cv2.imread("grid.jpg")
points = []

def click(event, x, y, flags, param):
    global img
    if event == cv2.EVENT_LBUTTONDOWN:
        X, Y = pixel_to_world(x, y)
        print(f"pixel=({x},{y}) -> world=({X:.2f}, {Y:.2f}) cm")

        cv2.circle(img, (x, y), 4, (0, 0, 255), -1)
        text = f"({X:.1f},{Y:.1f})"
        cv2.putText(img, text, (x + 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.imshow("test", img)

cv2.imshow("test", img)
cv2.setMouseCallback("test", click)
cv2.waitKey(0)
cv2.destroyAllWindows()