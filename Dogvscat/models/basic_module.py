import time

import torch as t


class BassicModule(t.nn.Module):
    """
    封装了nn.Module 主要提供 save和load两个方法
    """

    def __init__(self):
        super(BassicModule, self).__init__()
        self.model_name = str(type(self))  # 模型的默认名字

    def load(self, path):
        """
        加载指定路径的模型
        """
        self.load_state_dict(t.load(path))

    def save(self, name=None):
        """
        保存模型，默认使用“模型名字+时间”作为文件名
        如AlexNet_0710_23:57:29.pth
        """
        if name is None:
            prefix = "checkpoints/" + self.model_name + "_"
            name = time.strftime(prefix + "%m%d_%H:%M:%S.pth")
        t.save(self.state_dict(), name)
        return name

    def get_optimizer(self, lr, weight_decay):  # 优化器优化算法
        return t.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

class Flat(t.nn.Module):
    """
    把输入的reshape成（batch_size,dim_length)
    """

    def __init__(self):
        super(Flat, self).__init__()
        #self.size = size

    def forward(self, x):
        # 转换shape
        return x.view(x.size(0), -1)
