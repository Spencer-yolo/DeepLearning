import csv
import os.path

from Demos.rastest import val
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchnet.meter as meter
import models
from Dogvscat.data.dataset import DogCat
from Dogvscat.utils.visualizer import Visualizer
from config import DefaultConfig
import torch as t

opt = DefaultConfig()


def train(**kwargs):

    # 根据命令行参数更新配置
    opt.parse(**kwargs)
    vis = Visualizer(opt.env)   # 可视化操作visdom

    # step1:模型
    model = getattr(models, opt.model)()
    if opt.load_model_path:
        model.load(opt.load_model_path)
    if opt.use_gpu:
        model.cuda()

    # step2:数据
    train_data = DogCat(opt.train_data_root, train=True)
    val_data = DogCat(opt.test_data_root, train=False)
    train_dataloader = DataLoader(train_data, opt.batch_size, shuffle=True,
                                 num_workers=opt.num_workers)   # num_workers个woker大了 找寻batch快
    val_dataloader = DataLoader(val_data, opt.batch_size, shuffle=False,
                                num_workers=opt.num_workers)  # num_workers一般设置cpu核心数，如果cpu强可以大写

    # step3:目标函数和优化器
    criterion = t.nn.CrossEntropyLoss()   # 损失函数 交叉熵 计算期望与数据的距离，分类时好用
    lr = opt.lr
    optimizer = t.optim.Adam(model.parameters(),
                             lr = lr,
                             weight_decay = opt.weight_decay)

    # step4: 统计指标：平滑处理之后的损失，还有混淆矩阵（计算分类类别的准确率）
    loss_meter = meter.AverageValueMeter()   # 创建一个类 添加单值数据 然后进行取平均和方差计算
    confusion_matrix = meter.ConfusionMeter(2)
    previous_loss = 1e100

    # 训练
    for epoch in range(opt.max_epoch):

        loss_meter.reset()   # 清空序列
        confusion_matrix.reset()

        for ii,(data, label) in enumerate(train_dataloader): # 同时输出数据和数据的下标

            # 训练模型参数
            input = Variable(data)
            target = Variable(label)
            if opt.use_gpu:
                input = input.cuda()
                target = target.cuda()
            optimizer.zero_grad()
            score = model(input)
            loss = criterion(score,  target)
            loss.backward()   # 清楚loss
            optimizer.step()

            # 更新统计指标以及可视化
            loss_meter.add(loss.data[0])  # 添加数据每一次就只有一个
            confusion_matrix.add(score.data, target.data)

            if ii%opt.print_frep==opt.print_frep-1:
                vis.plot("loss", loss_meter.value()[0])

                # 如果需要，进入debug模式
                if os.path.exists(opt.debug_file):  # 如果该文件存在
                    import ipdb
                    ipdb.set_trace()
        model.save()

        # 计算验证集上的指标及可视化
        val_cm, val_accuracy = val(model, val_dataloader)
        vis.plot("val_accuracy", val_accuracy)
        vis.log(f"epoch:{epoch},lr:{lr},loss:{loss_meter.value()[0]},train_cm:{str(confusion_matrix.value())},val_cm{str(val_cm.value())}")

        # 如果损失不再下降，则降低学习率
        if loss_meter.value()[0] > previous_loss:
            lr = lr*opt.lr_decay
            for param_group in optimizer.param_groups:  # 长度为2的List,元素是两个字典
                param_group["lr"] = lr  # 更新优化器的学习率参数

        previous_loss = loss_meter.value()[0]

def eval(model, dataloader):
    """
    计算模型在验证集上的准确率等消息
    """

    # 把模型设为验证模式
    model.eval()

    confusion_matrix = meter.ConfusionMeter(2)
    for ii, data in enumerate(dataloader):
        input, label = data
        val_input = Variable(input, valatile=True)
        val_label = Variable(label.long(), volatile=True)
        if opt.use_gpu:
            val_input = val_input.cuda()
            val_label = val_label.cuda()
        score = model(val_input)
        confusion_matrix.add(score.data.squeeze(), label.long()) # 将label转换为Long类型的

        # 把模型恢复为训练模式
        model.train()

        cm_value = confusion_matrix.value()
        accuracy = 100.*(cm_value[0][0]+cm_value[1][1]/
                         (cm_value.sum()))
        return confusion_matrix, accuracy

def test(**kwargs):
    opt.parse(kwargs)

    # 模型
    model = getattr(models, opt.model)().eval()  # 在models寻找属性opt.model并转化为测试数据
    if opt.load_model_path:
        model.load(opt.load_model_path)
    if opt.use_gpu:
        model.cuda()

    # 数据
    train = DogCat(opt.test_data_root, test=True)
    test_dataloader = DataLoader(train,batch_size=opt.batch_size,shuffle=False,
                                 num_workers=opt.num_workers)

    results = []
    for ii,(data,label) in enumerate(test_dataloader):
        input = t.autograd.Variable(data, volatile=True)  # 将tensor包装成variable数据---一个data一个维度
        if opt.use_gpu:
            input = input.cuda()
        score = model(input)
        probability = t.nn.functional.softmax(score)[:,1].data.tolist()
        batch_results = [(path_,probability_)
                        for path_, probability_ in zip(path,probability)]
        results+=batch_results
    with opt.result_file:
        writer = csv.writer(opt.result_file)
        writer.writerow(results)
    return results

def help():
    """
    打印帮助的信息：python file.py help
    """
    print("""
        usage : python {0} <function> [--args=value,]
        <function> := train | test | help
        example: 
                python {0} train --env='env0701' --lr=0.01
                python {0} test --dataset='path/to/dataset/root/'
                python {0} help
        avaiable args:""".format(__file__))

    from inspect import getsource
    source = (getsource(opt.__class__))
    print(source)
