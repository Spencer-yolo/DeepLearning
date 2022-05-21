from torchvision import transforms
import cv2
from torch.utils.tensorboard import SummaryWriter

# transforms 工具箱  里面定义了很多类
# transforms.ToTensor 转换成tensor数据类型

writer = SummaryWriter("logs")

# 使用transforms
img_path = "data/train/cats/3.jpg"
img = cv2.imread(img_path)  #(W, H, C)
tensor_train = transforms.ToTensor()   # 返回一个类 对象
tensor_img = tensor_train(img)
print(tensor_img)