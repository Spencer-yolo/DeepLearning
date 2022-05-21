
# 方式1
import torch
import torchvision

torch.load("E:\\pycharm\\datasets/vgg16_method1.pth")



# 方式2加载
vgg16 = torchvision.models.vgg16(pretrained=False)
vgg16.load_state_dict(torch.load("路径"))   # 通过这种方式回复原来的模型结构

# model = torch.load("路径")
print(vgg16)