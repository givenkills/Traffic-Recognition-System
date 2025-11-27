import cv2
import numpy as np

# 定义颜色范围，这里使用HSV颜色空间进行更好的颜色识别
lower_left_turn = np.array([10, 150, 150])  # 左转灯的黄色范围（HSV）
upper_left_turn = np.array([40, 255, 255])

lower_right_turn = np.array([160, 150, 150])  # 右转灯的红色范围（HSV）
upper_right_turn = np.array([180, 255, 255])

# 打开视频文件
cap = cv2.VideoCapture(r"C:\项目4交通识别素材\12月19日.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 转换到HSV颜色空间
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 使用HSV阈值进行颜色筛选，检测左右转向灯
    mask_left = cv2.inRange(hsv, lower_left_turn, upper_left_turn)
    mask_right = cv2.inRange(hsv, lower_right_turn, upper_right_turn)

    # 进行形态学操作以去除噪点
    kernel = np.ones((5, 5), np.uint8)
    mask_left = cv2.morphologyEx(mask_left, cv2.MORPH_CLOSE, kernel)
    mask_right = cv2.morphologyEx(mask_right, cv2.MORPH_CLOSE, kernel)

    # 查找左转灯的位置
    contours_left, _ = cv2.findContours(mask_left, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_right, _ = cv2.findContours(mask_right, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 绘制左转灯的检测框
    for contour in contours_left:
        if cv2.contourArea(contour) > 500:  # 过滤掉太小的噪声
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)  # 黄色矩形框

    # 绘制右转灯的检测框
    for contour in contours_right:
        if cv2.contourArea(contour) > 500:  # 过滤掉太小的噪声
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)  # 红色矩形框

    # 显示结果
    cv2.imshow('Frame', frame)

    # 按 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()