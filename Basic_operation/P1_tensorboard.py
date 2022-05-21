# tansorboard 是 pytorch下的可视化工具  能可视化 标量 向量 音频 视频等等
from torch.utils.tensorboard import SummaryWriter
import cv2

writer = SummaryWriter("logs")   # 里面是事件名称

# writer.add_image()


# y=x   # add_scalar 使用
for i in range(100):
    writer.add_scalar("y = 2x", 2*i, i)  # 标题 y轴 x轴
                                        # 用命令行 tensorboard --logdir=logs（事件文件名） 显示图像 加入端口 --port= 端口

# add_image()使用 常用来观察训练结果  观察不同阶段的一些形式
img_path = "data/train/cats/1.jpg"
img2_path = "data/train/dogs/1.jpg"
img = cv2.imread(img_path)   # cv2读取照片（W, H, C)格式
img2 = cv2.imread(img2_path)

writer.add_image("img", img, 1, dataformats="HWC")    # 1是第几步 最后一个参数 （宽，高，通道数）这样形式时候用 默认（3，H, W)
writer.add_image("img", img2, 2, dataformats="HWC")
writer.close()