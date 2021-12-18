from NET import ResNet
from data_preprocess import MyDataset
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from PIL import Image

img_file1 = r'D:\py-project\fingerprint identification\FVC2002_DB1_A\1_1.jpg'
img_file2 = r'D:\py-project\fingerprint identification\FVC2002_DB1_A\1_2.jpg'

val_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5))
])
val_path = 'FVC2002_DB1_B'
val_dataset = MyDataset(val_path, transforms=val_transforms, mode='val')
val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False)

network = ResNet()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # 在N卡的cuda上运行，如果没有cuda就在CPU上运行
network = network.to(device)
network.load_state_dict(torch.load('model.pth'))
img1 = Image.open(img_file1).convert('L')
img2 = Image.open(img_file2).convert('L')
img1 = val_transforms(img1)
img2 = val_transforms(img2)
img = torch.cat([img1, img2], 0)
img = img.to(device)
img = img.unsqueeze(0)
y_pred = network(img)
_, preds_classes = torch.max(y_pred, 1)
if torch.Tensor.cpu(preds_classes) == 1:
    print('是同个人的指纹')
else:
    print('是不同人的指纹')
