import cv2
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from PIL import Image
img = cv2.imread("data/train/cats/1.jpg")
img_PIL = Image.open("data/train/dogs/1.jpg")

# Totensor 的一个使用
writer = SummaryWriter("logs")
trans_totensor = transforms.ToTensor()   # 定义一个totensor的类
img_tensor = trans_totensor(img)

writer.add_image("ToTensor", img_tensor)



# Normalize
# ``output[channel] = (input[channel] - mean[channel]) / std[channel]``
print(img_tensor[0][0][0])
trans_norm = transforms.Normalize([0.1, 0.2, 0.3], [0.1, 0.2, 0.3])
img_norm = trans_norm(img_tensor)
print(img_norm[0][0][0])
writer.add_image("Normalize", img_norm)

# Resize
print(img_PIL.size)
trans_resize = transforms.Resize((512,512))   # 定义一个Resize的类
# img PIL -> resize -> img_resize PIL
img_resize = trans_resize(img_PIL)
# img_resie PIL -> totensor -> img_resize tensor
img_resize = trans_totensor(img_resize)
print(img_resize)
writer.add_image("Resize", img_resize, 0)

# Compose - resize - 2
trans_resize_2 = transforms.Resize(512)
trans_compose = transforms.Compose([trans_resize_2, trans_totensor])
img_resize_2 = trans_compose(img_PIL)
writer.add_image("Resize", img_resize_2, 1)

# RandomCap   随机裁剪
trans_random = transforms.RandomCrop(210)   # RandomCrop类  裁剪210*210 的区域
trans_compose_2 = transforms.Compose([trans_random, trans_totensor])  # 列表中第一个参数 就是第一个操作 第二个参数第二个操作 调用
for i  in range(10):
    img_crop = trans_compose_2(img_PIL)
    writer.add_image("RandomCap", img_crop, i)




writer.close()