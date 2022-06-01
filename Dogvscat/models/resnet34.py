from torch import nn

from .basic_module import BassicModule
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    """
    实现子module:Residual Block
    """

    def __init__(self, inchannel, outchannel, stride=1, shortcut=None):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 3, stride, 1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1,bias=False),
            nn.BatchNorm2d(outchannel),   # 数据归一化操作

        )
        self.right = shortcut

    def forward(self,x):
        out = self.left(x)
        residual = x if self.right is None else self.right(x)
        out += residual
        return F.relu(out)   # 激活函数


class ResNet34(BassicModule):
    """
    实现主module:ResNet34
    ResNet34包含多个Layer,每个Layer又包含多个Residual block
    用子module来实现Residual block,用_make_layer函数来实现layer
    """

    def __init__(self, num_classes=2):
        super(ResNet34, self).__init__()
        self.model_name = "resnet34"

        # 前几层：图像转换
        self.pre = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=64,kernel_size=7,stride=2,
                      padding=3, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # 重复的layer,分别有3，4，6，3个residual block
        self.layer1 = self._make_layer(64, 128,3)
        self.layer2 = self._make_layer(128,256,4,stride=2)
        self.layer3 = self._make_layer(256,512,6,stride=2)
        self.layer4 = self._make_layer(512,512,3,stride=2)

        # 分类用的全连接
        self.fc = nn.Linear(in_features=512, out_features=num_classes)

    def _make_layer(self, inchannel, outchannel, block_num, stride=1):
        """
        构建layer,包含多个residual block 残缺快
        """
        shortcut = nn.Sequential(
            nn.Conv2d(in_channels=inchannel, out_channels=outchannel,
                      kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        layers=[]
        layers.append(ResidualBlock(inchannel, outchannel, stride, shortcut))

        for i in range(1, block_num):
            layers.append(ResidualBlock(outchannel,outchannel))
        return nn.Sequential(*layers)  # *layer-输出列表中的所有元素

    def forward(self, x):
        x = self.pre(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = F.avg_pool2d(x, 7)  # 均值池化 池化层就是减少参数量的
        x = x.view(x.size(0), -1)
        return self.fc(x)

