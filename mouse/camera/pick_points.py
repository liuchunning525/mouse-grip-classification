import cv2

points = []

def click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print("pixel:", x, y)
        points.append((x,y))

img = cv2.imread("grid.jpg")

cv2.imshow("grid", img)
cv2.setMouseCallback("grid", click)

cv2.waitKey(0)
cv2.destroyAllWindows()

print("Selected points:", points)