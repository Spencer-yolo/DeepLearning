import torchvision
from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import DataLoader
# 准备测试数据集
test_data = torchvision.datasets.CIFAR10("E:\\pycharm\\datasets", train=False, transform=torchvision.transforms.ToTensor())

# batch_size 每取几个数据打包一次   drop_last为True 不能打包一个batch_size就舍去  shuffle=True随机抓取数据
test_loader = DataLoader(dataset=test_data, batch_size=4, shuffle=True, num_workers=0, drop_last=False)

# 测试数据集中第一张图片集target
img, target = test_data[0]
print(img.shape)
print(target)

writer = SummaryWriter("dataloader")
step = 0
for data in test_loader:
    imgs, targets = data
    # print(img.shape)
    # print(target)
    writer.add_images("test_data", imgs, step)   # 注意是add_images 一次添加多个图片
    step = step + 1

writer.close()