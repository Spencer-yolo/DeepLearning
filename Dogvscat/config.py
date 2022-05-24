import warnings


class DefaultConfig(object):
    env = "default" # visdom环境名
    model = "AlexNet"  # 使用的模型，名字必须与models/__init__.py中的名字一致

    train_data_root = "./data/train/"
    test_data_root = "./data/test1"
    load_model_path = "checkpoints/model.pth"  # 加载预训练的模型的路径，为None代表不加载

    batch_size = 128
    use_gpu = True
    num_workers = 3  # how many workers for loading data
    print_frep = 20 # print info every N batch

    debug_file = "/tmp/debug" # if os.paht.exists(debug_file): enter ipdb
    result_file = "result.csv"

    max_epoch = 10  # 最大轮数
    lr = 0.1 # initial learning rate
    lr_decay = 0.95 # when val_loss increase, lr = lr*lr_decay
    weight_decay = 1e-4 # 损失函数  权重衰变

    def parse(self, kwargs):
        """
        根据字典kwargs 更新config参数
        """
        # 更新配置参数
        for k, v in kwargs.items():
            if not hasattr(self, k): # obkect:self name:k 如果该对象有该属性k返回True
                warnings.warn("Warning:opt has not attribut %s" %k)
            setattr(self, k, v)   # 重新设置属性值 属性k  值value

        # 打印配置信息
        print("user config:")
        for k,v in self.__class__.__dict__.items():
            if not k.startswith("__"):
                print(k, getattr(self, k))  # getattr 获取属性值


# 在主程序中使用
# 前面的
"""import models
from config import DefaultConfig

opt = DefaultConfig()
lr = opt.lr
model = getattr(models, opt.model)
dataset = DogCat(opt.train_data_root)"""

# 利用parse修改参数
"""opt = DefaultConfig()
new_config = {'lr':0.1,'use_gpu':False}
opt.parse(new_config)
opt.lr == 0.1"""

