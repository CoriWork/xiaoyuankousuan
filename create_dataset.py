from PIL import Image, ImageDraw, ImageFont
import os
import numpy as np
import random

# 创建数据集目录
os.makedirs('dataset', exist_ok=True)

# 字体路径
font_path = "./Roboto-Regular.ttf"
font_size = 28

# 初始化存储灰度值和标签的列表
images_data = []
labels = []

def generate_numbers(graph_num, num_list, images_data, labels, threshold=128):
    for i in range(graph_num):
        size = 50
        image = Image.new('L', (size, size), 255)  # 灰度图像，白色背景
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype(font_path, font_size)
        
        # 在图片中绘制数字
        random_num = np.random.choice(num_list)
        text = str(random_num)
        bbox = draw.textbbox((0, 0), text, font=font)  # 获取文本边界框
        text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]

        # 水平方向（x轴）随机生成位置，但确保数字不会超出边界
        max_x_offset = size - text_width
        x_offset = np.random.randint(0, max_x_offset + 1) if max_x_offset > 0 else 0

        # 垂直方向（y轴）在底部
        y_offset = size - text_height - 6

        position = (x_offset, y_offset)
        draw.text(position, text, fill=0, font=font)  # 使用黑色绘制文字
        
        # 将图片转换为 NumPy 数组
        image_array = np.array(image)

        binary_image = (image_array > threshold).astype(np.float32) * 255

        # 将二值图像保存为文件，以验证二值化效果
        binary_image_pil = Image.fromarray(binary_image).convert('L')
        binary_image_pil.save(f"binary_images/{random_num}.png")
        
        # 保存二值化后的数据
        images_data.append(binary_image)
        labels.append(random_num)
        
        if i % (graph_num // 10) == 0:
            print(f"Generating {i / graph_num * 100}%")
    print("Finish!")

# 基础训练：生成 0 到 20 的数字灰度图片
graph_num = 10000
generate_numbers(graph_num, list(range(0, 11)), images_data, labels)
# 专项训练：10几
graph_num = 200000
generate_numbers(graph_num, list(range(10, 21)), images_data, labels)
# # 专项训练：5 6 7 2
# graph_num = 5000
# generate_numbers(graph_num, [5, 6, 7, 2], images_data, labels)


combined = list(zip(images_data, labels))
random.shuffle(combined)
shuffled_images, shuffled_labels = zip(*combined)
images_data = list(shuffled_images)
labels = list(shuffled_labels)
# 将图像数据和标签保存为 .npy 文件
np.save('dataset/images_data.npy', np.array(images_data))
np.save('dataset/labels.npy', np.array(labels))
