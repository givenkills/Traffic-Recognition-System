from ultralytics import YOLO
import cv2

# 加载训练好的YOLO模型
model = YOLO("yolov8n.pt")  # 这里使用训练后的模型文件

# 打开视频文件
cap = cv2.VideoCapture(r"C:\项目4交通识别素材\12月23日.mp4")  # 替换为视频路径

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 使用YOLO进行推理
    results = model(frame)

    # 确保results是一个有效的Results对象
    if results:
        # 打印检测结果
        print(results[0].verbose())  # 使用verbose()来输出检测日志

        # 渲染检测框
        annotated_frame = results[0].plot()

        # 显示带检测框的图像
        cv2.imshow("Detected Lane", annotated_frame)

    # 按 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()