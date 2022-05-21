import torch
import torchvision
from PIL import Image
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear

img_path = "imgs\\dog.jpg"
dev = torch.device("cuda")

image = Image.open(img_path)
print(image)


transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32,32)),
                                            torchvision.transforms.ToTensor()])

image = transform(image)
print(image.shape)

class Simon(nn.Module):
    def __init__(self):
        super(Simon, self).__init__()
        self.model = Sequential(
            Conv2d(3, 32, 5, padding=2, stride=1),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2, stride=1),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2, stride=1),
            MaxPool2d(2),
            Flatten(),
            Linear(64*4*4, 64),
            Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x


model = torch.load("D:\\01-programming\\model_gpu")
print(model)
image = torch.reshape(image, (1, 3, 32, 32))
image = image.to(dev) # 因为是在GPU上训练的模型  所以要将输入转换成GPU的格式才行
model.eval() # 转换成测试类型
model = model.to(dev)
with torch.no_grad():
    output = model(image)

print(output.sum(axis=0))

