import torch
from torch.backends import cudnn

flag = torch.cuda.is_available()
if flag:
    print("cuda可以使用")
else:
    print("cuda不可以使用")

ngpu=1
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu>0) else "cpu")
print("驱动为",device)
print("GPU型号为",torch.cuda.get_device_name(0))

print(cudnn.is_available())

print("torch版本", torch.__version__)
print("cuda版本", torch.version.cuda)
print("cudnn版本", torch.backends.cudnn.version())
