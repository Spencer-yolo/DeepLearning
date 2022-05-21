import torch
import torchvision
from torch.utils.data import DataLoader
from torch import nn
from torch.nn import Conv2d
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("E:\\pycharm\\datasets", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=False)
dataloader = DataLoader(dataset, batch_size=64)  # batch_size 就是多少个图片分为一组

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=6, kernel_size=(3,3))  # 输入通道3个 输出通道6个

    def forward(self, x):
        x = self.conv1(x)
        return x

tudui = Tudui()

flag = 0
writer = SummaryWriter("logs")
for data in dataloader:
    imgs, targets = data
    output = tudui(imgs)
    # torch.Size([16, 3, 32, 32]) imgs.shape
    # torch.Size([16, 6, 30, 30])  output.shape
    output = torch.reshape(output, [-1, 3, 30, 30])
    writer.add_images("imgs", imgs, flag)
    writer.add_images("outputs", output, flag)
    flag +=1

writer.close()




