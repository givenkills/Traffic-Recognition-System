import cv2
import pytesseract
import numpy as np

# 打开视频文件或摄像头
video_path = r"C:\项目4交通识别素材\1.【超清】法国自驾游(第一视角)｜在法国普罗旺斯-阿尔卑斯-蓝色海岸大区-昂蒂(Av427878751,P1).mp4"
# 你的视频文件路径
cap = cv2.VideoCapture(video_path)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 转为灰度图像
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 使用边缘检测或二值化等方法来提高车牌检测效果
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    # 使用形态学操作增强车牌区域的检测效果（可选）
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(binary, kernel, iterations=1)

    # 查找图像中的轮廓
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # 过滤出矩形区域作为可能的车牌区域
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h
        if 2 < aspect_ratio < 6:  # 假设车牌长宽比大致为2到6
            # 画出可能的车牌区域轮廓
            cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)  # 绘制轮廓
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # 画出矩形框
            plate_region = frame[y:y + h, x:x + w]

            # 预处理车牌区域（如灰度化，二值化等）
            plate_gray = cv2.cvtColor(plate_region, cv2.COLOR_BGR2GRAY)
            _, plate_binary = cv2.threshold(plate_gray, 150, 255, cv2.THRESH_BINARY)

            # 使用 pytesseract 进行车牌字符识别
            plate_text = pytesseract.image_to_string(plate_binary, config='--psm 8')  # PSM 8: 适合单行文本
            plate_text = plate_text.strip()

            if plate_text:
                print(f"识别到车牌号: {plate_text}")

    # 显示处理后的帧
    cv2.imshow('Frame', frame)

    # 按键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放视频捕捉对象并关闭所有窗口
cap.release()
cv2.destroyAllWindows()