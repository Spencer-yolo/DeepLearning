
import torchvision

# 现有网络模型的使用及修改
# train_data = torchvision.datasets.ImageNet("E:\\pycharm\\datasets", split="train", download=False,
#                                            transform=torchvision.transforms.ToTensor())
from torch import nn

vgg16_true = torchvision.models.vgg16(pretrained=True)
print(vgg16_true)

vgg16_true.add_module("add_linear", nn.Linear(1000, 10))

vgg16_true.classifier.add_module("add2_linear", nn.Linear(1000, 10))  # 在classifier上添加层数
vgg16_true.classifier[1].add_module("add3_linear", nn.Linear(1000, 10))  # 修改classifier上的层数

