from torch import nn
from torchvision.models import squeezenet1_1
from torch.optim import Adam
from basic_module import BassicModule

class SqueezeNet(BassicModule):
    def __init__(self, num_classes=2):
        super(SqueezeNet, self).__init__()
        self.model_name = "squeezenet"
        self.model = squeezenet1_1(pretrained=True)  # 预训练的模型
        # 修改 原始的num_class :预训练模型是1000分类
        self.model.num_classes = num_classes
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.5),  # 随机删除几个输入神经元  防止过拟合，省时
            nn.Conv2d(512,num_classes,1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(13, stride=1)
        )

    def forward(self, x):
        return self.model(x)

    def get_optimizer(self, lr, weight_decay):
        # 因为使用了预训练模型，我们只要需要训练后面的分卸
        # 前面的特征提取部分可以保持不变
        return Adam(self.model.classifier.parameters(), lr,weight_decay=weight_decay)

