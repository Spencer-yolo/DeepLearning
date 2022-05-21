import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential

class Simon(nn.Module):
    def __init__(self):
        super(Simon, self).__init__()
        self.model = Sequential(
            Conv2d(3, 32, (5, 5), padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, (5, 5), padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, (5, 5), padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x

if __name__ == '__main__':
    simon = Simon()
    input = torch.ones((63, 3, 32, 32))
    output = simon(input)
    print(output)