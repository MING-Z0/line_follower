import torch
import torchvision.models as models

# 加载模型权重
model = models.resnet18()
model.fc = torch.nn.Linear(512, 2)
model.load_state_dict(torch.load(r'.\training\model\best_model.pth', map_location=torch.device('cpu')))  # 在此处使用 'cpu'，因为ONNX转换不需要 GPU
model.eval()

# 定义一个示例输入
example_input = torch.randn(1, 3, 224, 224)  # 输入大小为(1, 3, 224, 224)，假设输入通道数为3，图像大小为224x224

# 将模型转换为 ONNX 格式
onnx_output_path = r'.\training\model\resnet18.onnx'
torch.onnx.export(model,                      # 模型
                  example_input,             # 示例输入
                  onnx_output_path,          # 输出 ONNX 文件路径
                  input_names=['input'],     # 输入节点名称
                  output_names=['output'],   # 输出节点名称
                  export_params=True)        # 是否导出模型参数