import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset
import os
import re
from PIL import Image
from torch.utils.data import DataLoader
import pandas as pd
from NET import ResNet
from torch.autograd import Variable


class MyDataset(Dataset):
    def __init__(self, filepath, transforms=None, mode='train'):
        super(MyDataset, self).__init__()
        self.img_files, self.labels = self.load_data(filepath)
        self.transforms = transforms
        self.mode = mode
        self.filepath = filepath

    def __getitem__(self, index):
        img_file1_name, img_file2_name, label = self.img_files[index][0], self.img_files[index][1], self.labels[index]
        img_file1 = os.getcwd() + '\\' + self.filepath + '\\' + img_file1_name if self.mode == 'train' else os.getcwd() + '\\' + 'FVC2002_DB1_A' + '\\' + img_file1_name
        img_file2 = os.getcwd() + '\\' + self.filepath + '\\' + img_file2_name if self.mode == 'train' else os.getcwd() + '\\' + 'FVC2002_DB1_A' + '\\' + img_file2_name
        img1 = Image.open(img_file1).convert('L')
        img2 = Image.open(img_file2).convert('L')
        if self.transforms:  # 采用自定义的transform来实现图片的转换，使之可以直接进入神经网络运算
            img1 = self.transforms(img1)
            img2 = self.transforms(img2)
        img = torch.cat([img1, img2], 0)
        return img, label

    def __len__(self):
        return len(self.labels)

    def load_data(self, filepath):
        df = pd.read_csv(filepath + '.csv', index_col=None)
        img_file_names = np.array(df.iloc[:, :-1])
        labels = np.array(df.iloc[:, -1])
        return img_file_names, labels


def train(model, device, config):
    model = model.to(device)
    if os.path.exists(config['save_path']):  # 权重文件在不在
        model.load_state_dict(torch.load(config['save_path']))
        print('存在模型参数')
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(config['max_epoch']):
        for idx, data in enumerate(train_loader):
            x = data[0].to(device)
            y = data[1].to(device)
            model.train()
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            print('\repoch: {}\tindex: {}\tlr: {}\tloss: {}'.format(epoch, idx, config['lr'], loss), end='')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        torch.save(model.state_dict(), config['save_path'])
        print('保存模型')


def test(model, device, config):
    model = model.to(device)
    model.load_state_dict(torch.load(config['save_path']))
    correct_classes = 0
    Accuracy_list_classes = []
    for idx, data in enumerate(test_loader):
        x = data[0].to(device)
        y = data[1].to(device)
        y_pred = model(x)
        _, preds_classes = torch.max(y_pred, 1)
        correct_classes += torch.sum(preds_classes == y)
        current_accuracy = correct_classes.double() / (idx + 1) * 100
        # print(f'\rcorrect_classes: {correct_classes}\tcurrent_accuracy:{current_accuracy}\ty_pred:{y_pred}\ty:{y}\t',
        #       end='')
        print(f'\rcorrect_classes: {correct_classes}\tcurrent_accuracy:{current_accuracy}\ty_pred:{y_pred}\ty:{y}\t')
    epoch_acc_classes = correct_classes.double() / len(test_loader.dataset)
    Accuracy_list_classes.append(100 * epoch_acc_classes)
    print(f'accuracy:{epoch_acc_classes}')


config = {
    'max_epoch': 50,  # maximum number of epochs
    'batch_size': 32,  # mini-batch size for dataloader
    'lr': 0.0001,  # learning rate of SGD
    'early_stop': 100000,  # early stopping epochs (the number epochs since your model's last improvement)
    # 'accumulation_steps': 8,
    'save_path': 'model.pth',
}

if __name__ == '__main__':
    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),  # 依据概率水平翻转（默认0.5）
        transforms.RandomRotation(20),  # 随机旋转
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))  # 数据标准化，使其符合正态分布
    ])
    """如果是灰度图片使用transforms.Normalize((0.5), (0.5))，RGB图片使用transforms.Normalize((0.5，0.5，0.5), (0.5，0.5，0.5))"""

    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
    ])
    train_path = 'FVC2002_DB1_A'
    test_path = 'FVC2002_DB1_C'
    val_path = 'FVC2002_DB1_B'
    train_dataset = MyDataset(train_path, transforms=train_transforms, mode='train')
    test_dataset = MyDataset(test_path, transforms=test_transforms, mode='test')
    train_loader = DataLoader(dataset=train_dataset, batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # 在N卡的cuda上运行，如果没有cuda就在CPU上运行
    print('device确定')
    network = ResNet()

    train(network, device, config)

    test(network, device, config)
