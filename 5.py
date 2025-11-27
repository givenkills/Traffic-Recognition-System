import cv2
import pytesseract
from ultralytics import YOLO

# 加载YOLO模型
model = YOLO('yolov8n.pt')  # 或者使用训练好的车牌检测模型


# 车牌号识别函数
def recognize_license_plate(frame):
    # 使用YOLO进行推理
    results = model(frame)

    # 获取检测到的车牌
    for result in results:
        boxes = result.boxes  # 获取检测到的框
        for box in boxes:
            # 获取车牌区域的坐标
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # 车牌框的坐标

            # 从原图中裁剪出车牌区域
            license_plate_img = frame[y1:y2, x1:x2]

            # 使用Tesseract进行车牌号识别
            gray = cv2.cvtColor(license_plate_img, cv2.COLOR_BGR2GRAY)
            text = pytesseract.image_to_string(gray, config='--psm 8')  # --psm 8 用于单行字符识别
            print("Detected License Plate:", text.strip())


# 打开视频文件或摄像头
cap = cv2.VideoCapture(r"C:\项目4交通识别素材\12月23日.mp4")  # 替换为视频路径或使用摄像头

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    recognize_license_plate(frame)

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
