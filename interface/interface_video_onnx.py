import onnx
import onnxruntime
import numpy as np
import cv2
import torchvision.transforms as transforms

# 加载ONNX模型
onnx_model_path = r'.\training\model\resnet18.onnx'
onnx_session = onnxruntime.InferenceSession(onnx_model_path)

# 打开视频文件
video_path = r'.\data\video\VID_4.mp4'
cap = cv2.VideoCapture(video_path)

# 获取视频的宽度和高度
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 创建一个预处理图像的转换
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 创建视频写入对象
output_video_path = r'.\test\output\output_video_with_onnx_model.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, 20.0, (frame_width, frame_height))

# 对视频的每一帧进行推理和保存
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 对图像进行预处理
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = transform(image)
    image = np.expand_dims(image, axis=0)  # 添加 batch 维度

    # 进行推理
    inputs = {'input': image}
    outputs = onnx_session.run(None, inputs)

    # 将模型输出转换为图像坐标系的值
    def convert_to_image_coordinates(x, y, image_width, image_height):
        x_image = (x * 112 + 112) * image_width / 224
        y_image = (y * 112 + 112) * image_height / 224
        return x_image, y_image

    # 获取图像宽度和高度
    image_height, image_width, _ = frame.shape

    # 解析模型输出并转换为图像坐标系的值
    x_normalized, y_normalized = outputs[0][0]  # 假设第一个输出是预测的坐标
    x_image, y_image = convert_to_image_coordinates(x_normalized, y_normalized, image_width, image_height)

    # 在图像上绘制结果
    cv2.circle(frame, (int(x_image), int(y_image)), 5, (0, 255, 0), -1)

    # 将帧写入输出视频
    out.write(frame)

# 释放视频读取对象和视频写入对象
cap.release()
out.release()

print("Inference completed. Output video saved at:", output_video_path)
