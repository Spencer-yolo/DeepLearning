import torch
import torchvision
from torch import nn
from torch.nn import L1Loss, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Flatten, Linear

# 损失函数
from torch.utils.data import DataLoader

input = torch.tensor([1, 2, 3], dtype=torch.float32)
target = torch.tensor([1,2, 5], dtype=torch.float32)


loss1 = L1Loss(reduction="sum")  # 取和
loss2 = L1Loss(reduction="mean") # 取平均
result1 = loss1(input, target)
result2 = loss2(input, target)
print(result1)
print(result2)


# crossentropyloss  交叉商
x = torch.tensor([0.1, 0.2, 0.3])
y = torch.tensor([1])
x = torch.reshape(x, (1, 3))  # N=batch size c=number of classes 一行三列 行是实际的 列是推测的概率
loss_cross = CrossEntropyLoss()
result_cross = loss_cross(x, y)
print(result_cross) #利用公式计算

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
for data in dataloder:
    imgs, targets = data
    imgs = simon(imgs)
    result_cross2 = loss_cross2(imgs, targets) # 多对一
    result_cross2.backward()  # 自动计算所有的梯度
    print("ok")

