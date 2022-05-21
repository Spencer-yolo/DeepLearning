import torch
from torch import nn
from torch.nn import ReLU, Sigmoid  # ReLU <0 变为0 >0变为1
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

input = torch.tensor([[-1, 2],
                      [2, -10]])
input = torch.reshape(input, (-1, 1, 2, 2))
print(input.shape)


class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.relu1 = ReLU()   # 里面参数 True就是结果替换原数 False就是结果不替换原数输出为新值

    def forward(self, input):
        output = self.relu1(input)
        return output

tudui = Tudui()
output = tudui(input)
print(output)

class Simon(nn.Module):
    def __init__(self):
        super(Simon, self).__init__()
        self.sigmod1 = Sigmoid()   # 里面参数 True就是结果替换原数 False就是结果不替换原数输出为新值

    def forward(self, input):
        output = self.sigmod1(input)
        return output

simon = Simon()
writer = SummaryWriter("logs")
datasets = torchvision.datasets.CIFAR10("E:\\pycharm\\datasets", train=False, transform=torchvision.transforms.ToTensor())
dataloder = DataLoader(datasets, batch_size=64)
flag = 0
for data in dataloder:
    imgs, targets = data
    writer.add_images("origin", imgs, flag)
    output2 = simon(imgs)
    writer.add_images("Sigmoid", output2, flag)
    flag += 1

writer.close()
