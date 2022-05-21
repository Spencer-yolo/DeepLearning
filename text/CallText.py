import numpy as np
class Person:
    def __call__(self, name):
        print("__call__" + "hello" + name)

    def hello(self, name):
        print("hello" + name)

people = Person()
people("call")
people.hello("lisi")

y = np.random.randn(1,2,4)
print(y)