from torch.utils.data import DataLoader

import models
from Dogvscat.data.dataset import DogCat
from Dogvscat.utils.visualizer import Visualizer
from config import DefaultConfig

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
                                 num_workers=opt.num_workers)
    val_dataloader = DataLoader(val_data, opt.batch_size, shuffle=False,
                                num_workers=opt.num_workers)