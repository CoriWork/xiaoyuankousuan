import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os

# 检查是否有可用的 GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# 自定义数据集类
class NumberDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx].astype(np.float32) / 255.0  # 归一化
        label = self.labels[idx]
        image = torch.tensor(image).unsqueeze(0)  # 将图像维度从 (50, 50) 转换为 (1, 50, 50)
        return image, label

# 加载数据
images_data = np.load('dataset/images_data.npy')
labels = np.load('dataset/labels.npy')

# 创建数据集和数据加载器
dataset = NumberDataset(images_data, labels)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# 定义神经网络模型
class DigitRecognitionModel(nn.Module):
    def __init__(self):
        super(DigitRecognitionModel, self).__init__()
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

# 初始化模型、损失函数和优化器
model = DigitRecognitionModel().to(device)  # 将模型迁移到 GPU
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练函数
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, save_path='model.pth'):
    best_val_accuracy = 0.0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)  # 将数据迁移到 GPU
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)  # 将数据迁移到 GPU
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_accuracy = 100 * correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Train Loss: {running_loss/len(train_loader):.4f}, "
              f"Val Loss: {val_loss/len(val_loader):.4f}, "
              f"Val Accuracy: {val_accuracy:.2f}%")
        
        # 如果当前验证准确率更高，则保存模型
        if val_accuracy >= best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), save_path)
            print(f"Model saved with accuracy: {val_accuracy:.2f}%")

# 训练模型并保存
os.makedirs('saved_models', exist_ok=True)
train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, save_path='saved_models/best_model.pth')
