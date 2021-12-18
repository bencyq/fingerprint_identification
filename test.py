import torch
from PIL import Image
import torchvision.transforms as transforms
import os

x1_t = torch.normal(2 * torch.ones(100, 2), 1)
y1_t = torch.zeros(100)

x2_t = torch.normal(-2 * torch.ones(100, 2), 1)
y2_t = torch.ones(100)

x_t = torch.cat((x1_t, x2_t), 0)
y_t = torch.cat((y1_t, y2_t), 0)
image = Image.open(r'D:\py-project\fingerprint identification\FVC2002_DB1_A\1_7.jpg').convert('L')
train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # 依据概率水平翻转（默认0.5）
    transforms.RandomRotation(20),  # 随机旋转
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5))  # 数据标准化，使其符合正态分布
])
img = train_transforms(image)

