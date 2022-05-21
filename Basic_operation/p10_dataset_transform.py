import torchvision
from torch.utils.tensorboard import SummaryWriter

dataset_transform = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor()]
)
# train=True 训练集  False验证集  transforms参数转换数据类型
train_set = torchvision.datasets.CIFAR10(root="E:\\pycharm\\datasets", transform=dataset_transform, train=True)# 训练集
test_set = torchvision.datasets.CIFAR10(root="E:\\pycharm\\datasets", transform=dataset_transform, train=False)# 验证集


# print(test_set[0])
# print(test_set.classes)  # 标签
# img, target = test_set[0]
# print(img)   # PIL的图片数据类型
# print(target)
# img.show()

# print(test_set[1])  # 验证转换为tensor数据类型

writer = SummaryWriter("p10")  # 里面是要创建的储存文件名字
for i in range(10):
    img, target = test_set[i]
    writer.add_image("test_set", img, i)  # tensor 不用调dataforms


writer.close()


