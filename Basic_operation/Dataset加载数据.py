from torch.utils.data import Dataset
from PIL import Image
import os
import cv2

#reda_data
# 继承Dataset类
class MyData(Dataset):

    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.img_path = os.listdir(self.path)   # 获得数据集中图片的地址

    def __getitem__(self, idx):
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)
        img = cv2.imread(img_item_path)
        label = self.label_dir
        return img, label

    def __len__(self):
        return len(self.img_path)


root_dir = "C:\\Users\\33880\\Desktop\\DL\\data\\train"
cats_label_dir = "cats"
dogs_label_dir = "dogs"
cats_dataset = MyData(root_dir, cats_label_dir)
dogs_dataset = MyData(root_dir, dogs_label_dir)
print(r"数据个数", cats_dataset.__len__())
print("猫数据路径", cats_dataset.path)
"""img, label = cats_dataset[1]
cv2.imshow("cat1", img)
cv2.waitKey(0)"""

train_dataset = cats_dataset + dogs_dataset
print("猫数据集", len(cats_dataset))
print("狗数据集", len(dogs_dataset))
print("训练集", len(train_dataset))
