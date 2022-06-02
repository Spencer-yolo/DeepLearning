from torch import nn

from .basic_module import BassicModule  # 导入库前面加.表示导入当前文件夹内的一个包

class AlexNet(BassicModule):
    """
    code from torchvision/models/alexnet.py
    """

    def __init__(self, num_classes=2):
        super(AlexNet, self).__init__()

        self.model_name = "alexnet"

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11,stride=4,padding=2),
            nn.ReLU(inplace=True),   # inplace-更新后的是否覆盖原来的
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
     )
        self.classifier = nn.Sequential(
            nn.Dropout(),  # 防止过拟合-每一次的时候让一些隐藏元取值0 然后再恢复
            nn.Linear(256*6*6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256*6*6)   # 改变张量的shape 对数据直接操作不是拷贝操作
        x = self.classifier(x)
        return x