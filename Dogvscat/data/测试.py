

filename = "data/test1/8973.jpg"
print(filename.split(".")[-2].split("/"))
print(filename.split("."))
print(1 if "dog" in filename.split("/")[-1] else 0)


# # visdom测试
# import numpy as np
# import visdom
# viz = visdom.Visdom()
# viz.image(
#     np.random.rand(3, 512, 256), #随机生成一张图
#     opts = dict(title = 'Random!', caption = 'how random'),
# )


# return self测试
class Test:
    def __init__(self, name:str):
        self.name = name

    def testn(self, name):
        self.name = name
        return name

    def testn2(self, name):
        self.name = name
        return name

    def __repr__(self):
        return "repr" + self.name

test = Test("simon")
# print(test.testn("spencer").testn("simon"))   # 报错 新的test.testn("spencer")不是一个实例

class Test_self:
    def __init__(self, name):
        self.name =name

    def prin_1(self, name):
        self.name = name
        return self

    def prin_2(self, name):
        self.name = name
        return self

    def __repr__(self):
        return self.name

test_self = Test_self("simon_self")
print(test_self.prin_1("spencer_self").prin_2("selfceshi2"))  # 成功运行 test_self.prin_1("spencer_self") 是一个实例 包含实例的所有方法


#  hasattr函数的测试
class Hasattr:
    def __init__(self, name="simon", age = "19"):
        self.name = name
        self.age = age

test_hasattr = Hasattr()
print(hasattr(test_hasattr, "name"))

# 测试 获取类的属性hasattr()  和 修改属性setattr()
class Tom:
    name = "simon"
    age = "19"
    school = "hebeishida"

tom = Tom()
if hasattr(tom, "name"):
    setattr(Tom,"name", "Spencer")
print(tom.name)
print()