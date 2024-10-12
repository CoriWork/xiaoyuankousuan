import torch
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F

# 定义与训练时相同的神经网络模型结构
class NumberClassifier(nn.Module):
    def __init__(self):
        super(NumberClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)  # 卷积层1
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1) # 卷积层2
        self.pool = nn.MaxPool2d(2, 2)               # 最大池化层
        self.fc1 = nn.Linear(64 * 12 * 12, 128)      # 全连接层1
        self.fc2 = nn.Linear(128, 21)                # 全连接层2, 输出21类 (0-20)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 12 * 12)  # 展平
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 加载模型
model = NumberClassifier()
model.load_state_dict(torch.load('saved_models/best_model.pth'))
model.eval()  # 设置模型为评估模式
print("Load finished!")

# 定义图片预处理函数
def preprocess_image(image_path):
    image = Image.open(image_path).convert('L')  # 转换为灰度图像
    image = image.resize((50, 50))               # 调整大小到50x50
    image.save(image_path)
    image = np.array(image).astype(np.float32) / 255.0  # 归一化
    image = torch.tensor(image).unsqueeze(0).unsqueeze(0)  # (1, 1, 50, 50) 维度
    return image

# 定义识别函数
def predict_image(model, image_path):
    image = preprocess_image(image_path)
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        return predicted.item()

# 识别一组图片
image_paths = [f'{i}.png' for i in range(1, 10)]
predictions = [predict_image(model, path) for path in image_paths]

# 输出结果
for path, pred in zip(image_paths, predictions):
    print(f"Image: {path}, Predicted number: {pred}")
