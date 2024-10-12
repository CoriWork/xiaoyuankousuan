from PIL import Image, ImageDraw, ImageFont
import os
import numpy as np

# 字体路径，请确保路径正确
font_path = "./Avenir-Medium-6.otf"  # 替换为实际字体文件路径
font_size = 32

for i in range(1):
    size = 50
    target_width, target_height = 33, 25  # 目标宽度和高度
    image = Image.new('L', (size, size), 255)  # 灰度图像，白色背景
    draw = ImageDraw.Draw(image)
        
    # 尝试不同字体大小直到文本边界框符合要求
    font_size = 1
    while True:
        font = ImageFont.truetype(font_path, font_size)
        text = str(6)
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
        print(font_size, 1000)
        print(text_width, text_height)

        # 如果字体宽高接近目标宽高，就停止调整
        if text_width >= target_width and text_height >= target_height:
            break
        font_size += 1
