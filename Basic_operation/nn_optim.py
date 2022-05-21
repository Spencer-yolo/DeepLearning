import torch.optim
import torchvision
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear, CrossEntropyLoss
from torch.utils.data import DataLoader
# 优化器的使用
datasets = torchvision.datasets.CIFAR10("E:\\pycharm\\datasets", train=False, transform=torchvision.transforms.ToTensor())
dataloder = DataLoader(datasets, batch_size=1)
class Simon(nn.Module):
    def __init__(self):
        super(Simon, self).__init__()
        self.model1 = Sequential(
            Conv2d(3, 32, (5, 5), padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, (5, 5), padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, (5, 5), padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self, x):
        x = self.model1(x)
        return x
simon = Simon()
loss_cross2 = CrossEntropyLoss()
optim = torch.optim.SGD(simon.parameters(), lr=0.01)  # 创建一个优化器  # 变化幅度取决于 lr（学习率)
for epoch in range(20):   # 学习20次
    running_lose = 0.0
    for data in dataloder: # 每轮一个图片学习一次
        imgs, targets = data
        imgs = simon(imgs)
        result_cross2 = loss_cross2(imgs, targets) # 多对一
        optim.zero_grad()   # 将将梯度都清零 方面下一步重新计算梯度的值
        result_cross2.backward()  # 自动计算所有的梯度 反向传播
        optim.step()   # 开始优化
        running_lose = running_lose + result_cross2
    print(running_lose)