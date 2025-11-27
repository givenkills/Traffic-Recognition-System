import torch
import cv2
import numpy as np
from torchvision import models, transforms

# 加载预训练模型
model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# 预处理函数
def transform_image(image):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0)

# 视频捕获
cap = cv2.VideoCapture(r"C:\项目4交通识别素材\12月19日.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 转换为Tensor并进行推理
    tensor = transform_image(frame)
    with torch.no_grad():
        prediction = model(tensor)

    # 获取预测框
    boxes = prediction[0]['boxes'].cpu().numpy()
    labels = prediction[0]['labels'].cpu().numpy()
    scores = prediction[0]['scores'].cpu().numpy()

    # 可视化识别结果
    for i in range(len(boxes)):
        if scores[i] > 0.8:  # 设置阈值
            x1, y1, x2, y2 = boxes[i]
            label = labels[i]
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f"Label: {label} Score: {scores[i]:.2f}", (int(x1), int(y1)-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 显示视频
    cv2.imshow('Traffic Detection', frame)

    # 按'q'键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()