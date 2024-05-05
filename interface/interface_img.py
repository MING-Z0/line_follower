import torch
import cv2
import torchvision.models as models
import torchvision.transforms as transforms

# 加载模型权重
model = models.resnet18()
model.fc = torch.nn.Linear(512, 2)
model.load_state_dict(torch.load(r'.\training\model\best_model.pth', map_location=torch.device('cuda')))
model.eval()

# 读取图像
image_path = r'.\data\image_train\xy_342_368_62d8aa52-0934-11ef-b8ba-709cd13b16bd.jpg'  # 指定图像路径
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 对图像进行预处理
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
image = transform(image)
image = image.unsqueeze(0)  # 添加 batch 维度

# 进行推理
with torch.no_grad():
    outputs = model(image)

# 将模型输出转换为图像坐标系的值
def convert_to_image_coordinates(x, y, image_width, image_height):
    x_image = (x * 112 + 112) * image_width / 224
    y_image = (y * 112 + 112) * image_height / 224
    return x_image, y_image

# 获取图像宽度和高度
image_height, image_width, _ = cv2.imread(image_path).shape

# 解析模型输出并转换为图像坐标系的值
x_normalized, y_normalized = outputs.numpy()[0]
x_image, y_image = convert_to_image_coordinates(x_normalized, y_normalized, image_width, image_height)

# 输出转换后的图像坐标系的值
print("Normalized coordinates:", x_normalized, y_normalized)
print("Image coordinates:", x_image, y_image)
