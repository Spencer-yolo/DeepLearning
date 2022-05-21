import torchvision.datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import time
from model import *   # 从model文件中导入所有的参数


# 定义训练的设备 如果用cpu的话里面写cpu 用gpu的话写 cuda  单显卡 cuda 和cuda:0一样
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 有gpu用 没有就cpu

# 下载数据集
train_data = torchvision.datasets.CIFAR10("D:\\01-programming\\datasets", train=True,
                                          transform=torchvision.transforms.ToTensor())

text_data = torchvision.datasets.CIFAR10("D:\\01-programming\\datasets", train=False,
                                         transform=torchvision.transforms.ToTensor())

# 用Dataloader 加载数据集
train_loader = DataLoader(train_data, batch_size=64)
text_loader = DataLoader(text_data, batch_size=64)

simon = Simon()
simon = simon.to(device)

# 损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)

# 优化器  SGD随机梯度下降
# 1e-2 = 1*(10)^(-2)
learning_rate = 1e-2
optimizer = torch.optim.SGD(simon.parameters(), lr=learning_rate)

# 设置训练网络的一些参数
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_text_step = 0
# 训练的轮数
epoch = 1

start = time.time()
# 添加tensorboard
writer = SummaryWriter("D:\\logs")
for i in range(epoch):
    print(f"————————第{i}轮开始——————————")

    # 训练步骤开始
    simon.train()   # 只对某些指定模型有用利用dropout
    for data in train_loader:
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs = simon(imgs)
        loss = loss_fn(outputs, targets)

        # 优化器优化模型
        optimizer.zero_grad()  # 梯度清零
        loss.backward()      # 反向传播求梯度
        optimizer.step()

        total_train_step += 1

        if total_train_step % 100 == 0:
            print(f"训练次数: {total_train_step}, loss: {loss.item()}")   # 用loss.item()可以去掉外面的tensor直接是实数
            writer.add_scalar("train_loss", loss.item(), total_train_step)


    # 测试步骤
    simon.eval()
    total_text_lose = 0
    total_accurarcy = 0
    with torch.no_grad():  # 反向传播时不会自动求梯度防止一些优化
        for data in text_loader:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = simon(imgs)
            loss = loss_fn(outputs, targets)
            total_text_lose = total_text_lose + loss.item()
            accurarcy = ((outputs.argmax(1) == targets).sum())   # argmax 范围指定维度中最大值的位置 0列 1行 有几个True就是有几个正确的
            total_accurarcy = total_accurarcy + accurarcy
        print(f"整体测试题上的loss:{total_text_lose}")
        print(f"整体的正确率为: {total_accurarcy/len(text_data)}")   # 正确率多用于分类中
        writer.add_scalar("text_loss", total_text_lose, total_text_step)
        writer.add_scalar("text_accuracy", total_accurarcy/len(text_data), total_text_step)
        total_text_lose += 1

    torch.save(simon, "D:\\01-programming\\models")
    print("模型已保存")

end = time.time()
print(end - start)  # 第一轮用了9.8s
writer.close()