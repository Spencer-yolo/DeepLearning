import os

from PIL.Image import Image
from torch.utils import data
from torchvision import transforms as T


class DogCat(data.Dataset):
    def __init__(self, root, transform=None, train=True, test=False):
        """
        目标：获取所有图片地址，并根据训练，验证，测试划分数据
        :param root:  图片的根目录
        :param transform:
        :param train:
        :param test:
        """
        self.test = test
        imgs = [os.path.join(root, img) for img in os.listdir(root)]  # os.path.join连接两个或多个路径  os.listdir（return list filename from the directory)
        # test1: data/test1/8973.jpg  传入imgs的名字 (the format of the pictures)
        # train: data/train/cat.10004.jpn
        if self.test: # sorted-return a new list param-key 指定排序的元素
            imgs = sorted(imgs, key=lambda x: int(x.split(".")[-2].split("/")[-1]))
        else:   # split-return a list:the factors parted by ""  ["8973", "jpg"]
            imgs = sorted(imgs, key=lambda x: int(x.split(".")[-2]))

        imgs_num = len(imgs)

        # 划分测试，训练，验证集， 验证：训练 = 3：7
        if self.test:  # 测试集
            self.imgs = imgs
        elif train: #训练级
            self.imgs = imgs[:int(0.7*imgs_num)]  # 从第一个到 70 precent 张pictures
        else:  # 验证集
            self.imgs = imgs[:int(0.7*imgs_num)]

        if transform is None:

            # 数据转换操作，测试验证和训练的数据转换有所区别
            # x = x-mean(x)/std(x) # Normalize images with three channels
            normalize = T.Normalize(mean= [0.485, 0.456, 0.406],
                                   std= [0.299, 0.224, 0.225])
            # 测试集和验证集
            if self.test or not train:  # 是测试集 且有图片
                self.transforms = T.Compose([
                    T.Resize(224),  # logogram (224,225)
                    T.CenterCrop(224), # 中心裁剪
                    T.ToTensor(),  # tansform tensor farmate
                    normalize
                ])
            # 训练集
            else:
                self.transforms = T.Compose([
                    T.Resize(256),
                    T.RandomSizedCrop(224), #数据扩充（数据增强）的意义在于通过人为的随机范围裁剪，缩放，旋转等操作，增大训练集中的数据多样性、全面性，进而提高模型的泛化能力。
                    T.ToTensor(),
                    normalize
                ])

    def __getitem__(self, index): # 把dataset想像成一个列表 这个函数返回第index个样本的全部数据
        """
        返回一张图片的数据
        对于测试集，没有label,返回图片id,如1000.jpg 返回1000
        :param index:
        :return:
        """

        img_path = self.imgs[index]
        if self.test:
            label = int(self.imgs[index].split(".").split("/")[-1])  # 测试集
        else:  # 训练集
            label = 1 if "dog" in img_path.split("/")[-1] else 0  # 如果成立1 否则0
        data = Image.open(img_path)
        data = self.transforms(data)
        return data, label
    def __len__(self):
        """
        :return: 数据集中所有照片的个数
        """
        return len(self.imgs)



