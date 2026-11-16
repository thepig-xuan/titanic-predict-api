import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets,transforms

train_data = datasets.MNIST(root='./data',train=True,download=True,transform=transforms.ToTensor())
test_data = datasets.MNIST(root='./data',train=False,download=False,transform=transforms.ToTensor())


train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_data,  batch_size=64, shuffle=False)

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3)   # 1 通道 → 8 张特征图
        self.relu  = nn.ReLU()                        # 负数变 0
        self.pool  = nn.MaxPool2d(2)                  # 2×2 取最大，边长减半
        self.fc    = nn.Linear(8 * 13 * 13, 10)       # 1352 → 10

    def forward(self, x):
        x = self.conv1(x)    # [64,1,28,28] → [64,8,26,26]
        x = self.relu(x)     # 形状不变
        x = self.pool(x)     # [64,8,26,26] → [64,8,13,13]
        x = x.flatten(1)     # [64,8,13,13] → [64,1352]
        x = self.fc(x)       # [64,1352]   → [64,10]
        return x

model = CNN()
print(model)
print("参数总数:", sum(p.numel() for p in model.parameters()))