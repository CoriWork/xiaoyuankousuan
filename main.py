import torch
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import pyautogui
import time
import pygetwindow as gw
import keyboard
import os
from PIL import ImageGrab

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
threshold = 160
def preprocess_image(image, threshold=threshold):
    image = image.convert('L')  
    image = image.resize((50, 50))
    image_array = np.array(image)
    binary_image = (image_array > threshold).astype(np.float32) * 255
    binary_image = torch.tensor(binary_image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return binary_image

# 定义识别函数
def predict_image(model, image):
    image = preprocess_image(image)
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        return predicted.item()

def draw_greater_sign(start_x, start_y, size, duration):
    pyautogui.mouseDown(start_x, start_y)
    pyautogui.moveTo(start_x + size, start_y + size, duration=duration)
    pyautogui.moveTo(start_x, start_y + 2 * size, duration=duration)
    pyautogui.mouseUp()

def draw_less_sign(start_x, start_y, size, duration):
    pyautogui.mouseDown(start_x, start_y)
    pyautogui.moveTo(start_x - size, start_y + size, duration=duration)
    pyautogui.moveTo(start_x, start_y + 2 * size, duration=duration)
    pyautogui.mouseUp()

# 点击“开始PK”按钮
def start_pk(window_x, window_y, width, height):
    pyautogui.click(window_x + width * 0.39, window_y + height * 0.73)

# 画图形和点击后续按钮以开始下一把
def pk_and_next_pk(window_x, window_y, width, height):
    times = 40
    time.sleep(13.5)
    last_num1 = 0
    last_num2 = 0
    wrong_count = 0
    for i in range(times):
        bias_x = width / 2
        bias_y = height * 3 / 5
        size = 30
        duration = 0.01
        box1 = (window_x + 270, window_y + 354, window_x + 370, window_y + 454)
        box2 = (window_x + 470, window_y + 354, window_x + 570, window_y + 454)
        screenshot1 = ImageGrab.grab(box1)
        screenshot2 = ImageGrab.grab(box2)
        image = screenshot1.convert('L')
        image = image.resize((50, 50))
        image = np.array(image)
        binary_image = (image > threshold).astype(np.uint8) * 255
        binary_image_pil = Image.fromarray(binary_image).convert('L')
        binary_image_pil.save(f"main_images/1.png")

        image = screenshot2.convert('L')
        image = image.resize((50, 50))
        image = np.array(image)
        binary_image = (image > threshold).astype(np.uint8) * 255
        binary_image_pil = Image.fromarray(binary_image).convert('L')
        binary_image_pil.save(f"main_images/2.png")
        
        num1 = int(predict_image(model, screenshot1))
        num2 = int(predict_image(model, screenshot2))
        print(num1, num2)
        # 跳出持续错误识别
        if last_num1 == num1 and last_num2 == num2:
            wrong_count += 1
            time.sleep(0.7)
            if last_num1 < last_num2:
                if wrong_count < 2:
                    draw_greater_sign(window_x + bias_x, window_y + bias_y, size, duration)
                else:
                    draw_less_sign(window_x + bias_x, window_y + bias_y, size, duration)
            elif last_num1 > last_num2:
                if wrong_count < 2:
                    draw_less_sign(window_x + bias_x, window_y + bias_y, size, duration)
                else:
                    draw_greater_sign(window_x + bias_x, window_y + bias_y, size, duration)
            else:
                if i % 2 == 0:
                    draw_greater_sign(window_x + bias_x, window_y + bias_y, size, duration)
                else:
                    draw_less_sign(window_x + bias_x, window_y + bias_y, size, duration)
        else:
            wrong_count = 0
            last_num1 = num1
            last_num2 = num2
            if num1 > num2:
                draw_greater_sign(window_x + bias_x, window_y + bias_y, size, duration)
            else:
                draw_less_sign(window_x + bias_x, window_y + bias_y, size, duration)
        time.sleep(0.45)
    # 点击“开心收下”
    time.sleep(1)
    pyautogui.click(window_x + width / 2, window_y + height * 0.83)
    # 点击“继续”
    time.sleep(1)
    pyautogui.click(window_x + width * 5 / 6, window_y + height * 0.95)
    # 点击“继续PK”
    time.sleep(1)
    pyautogui.click(window_x + width / 2, window_y + height * 0.83)
    


def kill_process():
    print("空格键被按下，程序终止。")
    os._exit(0)  # 使用 os._exit() 来立即终止程序

def main():
    # 监听空格键按下事件
    keyboard.add_hotkey('space', kill_process)

    windows = gw.getWindowsWithTitle('小猿口算修罗场')
    for window in windows:
        print(f'窗口大小: {window.width} x {window.height}')
        print(f'窗口位置: ({window.left}, {window.top})')
        start_pk(window.left, window.top, window.width, window.height)
        for i in range(10):
            pk_and_next_pk(window.left, window.top, window.width, window.height)

if __name__ == "__main__":
    main()
