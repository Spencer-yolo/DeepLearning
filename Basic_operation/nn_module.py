import torch
from torch import nn

class Tudui(nn.Module): # 继承父类

    def __init__(self):
        super().__init__()   # 继承父类的init方法

    def forward(self, input):
        output = input+1
        return output

tudui = Tudui()
x = torch.tensor(1.0)  # 一维向量
output = tudui(x)
print(output)
