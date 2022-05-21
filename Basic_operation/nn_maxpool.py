import torch
import torchvision.datasets
from torch import nn
from torch.nn import MaxPool2d
# 最大池化 局部区域选取最大值 降低文件尺寸
# 对矩阵的
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

input = torch.tensor([[1, 2, 0, 3, 1],
                      [0, 1, 2, 3, 1],
                      [1, 2, 1, 0, 0],
                      [5, 2, 3, 1, 1],
                      [2, 1, 0, 1, 1]], dtype=torch.float32)

input = torch.reshape(input, (-1, 1, 5, 5))  # 池化的数据类型为（N,C,H,W)
print(input.shape)
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
    #    self.maxpool2 = MaxPool2d(kernel_size=3, ceil_mode=True)   # ceil_mode为True时候 保存数量不够kennel_size的
        self.maxpool2 = MaxPool2d(kernel_size=3, ceil_mode=False)    # ceil_mode为False 不保存数量不够的
    def forward(self, input):
        output = self.maxpool2(input)
        return output

tudui = Tudui()
print(tudui(input))

# 对图片数据集的
datasets = torchvision.datasets.CIFAR10("D:\\01-programming\\datasets", train=False,
                                        transform=torchvision.transforms.ToTensor())

dataloder = DataLoader(datasets, batch_size=64)
writer = SummaryWriter("logs")
flag = 0
for data in dataloder:
    imgs, targets = data
    output = tudui(imgs)
    writer.add_images("origin", imgs, flag)
    writer.add_images("maxpool", output, flag)
    flag+=1

writer.close()




