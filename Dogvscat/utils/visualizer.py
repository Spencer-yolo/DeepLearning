import time

import numpy as np
import visdom

# 工具函数
class Visualizer(object):
    """
    封装了visdom的基本操作，仍然可以通过self.vis.function或self.function调用原生的visdom接口
    比如
    self.text("hello visdom")
    self.historgram(t.randn(1000))
    self.line(t.arange(0,10), t.arange(1,11))
    """

    def __init__(self, env="default", **kwargs):  # **keargs-是传入字典形式的参数
        self.vis = visdom.Visdom(env=env, **kwargs)

        # 画的第几个数， 相当于横坐标
        # 保存（“loss",23) 即Loss的第23个点
        self.index = {}
        self.log_text = ""

    def reinit(self, env="default", **kwargs):
        """
        修改visdom的位置
        """
        self.vis = visdom.Visdom(env=env, **kwargs)
        return self   # 返回一个实例  让新的具有实例的性质 可以多个弄

    def plot_many(self, d):
        """
        一次plot多个
        :param d: dict(name,value) i.e. ("loss",0.11)
        """

    def img_many(self, d):
        for k, v in d.items():
            self.img(k, v)

    def plot(self, name, y, **kwargs):
        """
        self.plot("loss",1.00)
        """
        x = self.index.get(name, 0)  # 第二个param是默认的 没有找到时候的
        self.vis.line(Y=np.array([y]), X=np.array([x]),  # x,y轴数据
                      win=name,  # the name of windows
                      opts=dict(title=name), #
                      update=None if x==0 else "append",  # append以添加方式加入
                      **kwargs
                      )
        self.index[name] = x+1

    def img(self, name, img_, **kwargs):
        """
        self.img("input_img", t.Tensor(64,64)
        self.img('input_imgs", t.Tensor(3,64,64)
        self.img("input_imgs", t.Tensor(100, 1, 64, 64）
        self.img("input_imgs", t.Tensor(100,3,64,64), nrows=10)
        !!! don`t ~~self.img("input_imgs", t.Tensor(100,64,64),nrows=10)~~ !!!
        """
        self.vis.images(img_.cpu().numpy(),
                        win=name,
                        opts=dict(title=name),
                        **kwargs
                        )

    def log(self, info, win="log_text"):
        """
        self.log({"loss":1, "lr":0.0001})
        """
        self.log_text += ("[{time}] {info} <br}".format(
            time=time.strftime("%m%d_%H%M%S"),
            info=info
        ))
        self.vis.text(self.log_text, win)

    def __getattr__(self, name):
        """
        自定义的plot, image, log, plot_many等除外
        self.function 等价于self.vis.function
        """
        return getattr(self.vis, name)