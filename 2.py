import cv2
import numpy as np

def detect_lanes(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 转为灰度图
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)  # 高斯模糊
    edges = cv2.Canny(blurred, 50, 150)  # 边缘检测

    # 定义感兴趣区域
    height, width = edges.shape
    region_of_interest = np.array([[
        (0, height),
        (width // 2, height // 2),
        (width, height)
    ]], np.int32)
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, region_of_interest, 255)
    masked_edges = cv2.bitwise_and(edges, mask)

    # 霍夫变换检测车道线
    lines = cv2.HoughLinesP(masked_edges, 1, np.pi / 180, 120, minLineLength=50, maxLineGap=5)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

    return frame

# 视频捕获
cap = cv2.VideoCapture(r"C:\项目4交通识别素材\12月23日.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 车道线检测
    frame_with_lanes = detect_lanes(frame)

    # 显示视频
    cv2.imshow('Lane Detection', frame_with_lanes)

    # 按'q'键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()