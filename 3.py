import cv2
import numpy as np


def detect_traffic_lights(frame):
    # 转换到HSV颜色空间
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 定义红色、绿色、黄色的范围
    red_lower = np.array([0, 120, 70])
    red_upper = np.array([10, 255, 255])
    green_lower = np.array([35, 100, 100])
    green_upper = np.array([85, 255, 255])
    yellow_lower = np.array([20, 100, 100])
    yellow_upper = np.array([30, 255, 255])

    # 红灯检测
    red_mask = cv2.inRange(hsv, red_lower, red_upper)
    red_detected = np.any(red_mask)

    # 绿灯检测
    green_mask = cv2.inRange(hsv, green_lower, green_upper)
    green_detected = np.any(green_mask)

    # 黄灯检测
    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
    yellow_detected = np.any(yellow_mask)

    if red_detected:
        cv2.putText(frame, 'Red Light', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    elif green_detected:
        cv2.putText(frame, 'Green Light', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    elif yellow_detected:
        cv2.putText(frame, 'Yellow Light', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    return frame


# 视频捕获
cap = cv2.VideoCapture(r"C:\项目4交通识别素材\12月23日.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 交通灯识别
    frame_with_lights = detect_traffic_lights(frame)

    # 显示视频
    cv2.imshow('Traffic Light Detection', frame_with_lights)

    # 按'q'键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()