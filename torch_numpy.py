import torch
import numpy as np

#神经网络参数为Variable 变量形式计算
#在 Torch 中的 Variable 就是一个存放会变化的值的地理位置.
# 里面的值会不停的变化. 就像一个裝鸡蛋的篮子, 鸡蛋数会不停变动.
# 那谁是里面的鸡蛋呢, 自然就是 Torch 的 Tensor 咯.
# 如果用一个 Variable 进行计算, 那返回的也是一个同类型的 Variable.
from torch.autograd import Variable

"""
numpy torch 数据对比

np_data = np.arange(6).reshape(2, 3)
#torch_data = torch.arange(6)
torch_data = torch.from_numpy(np_data)

torch2array = torch_data.numpy()
print(
    "\n numpy_data", np_data,
    "\n torch_data", torch_data,
    "\n torch2array", torch2array,
)


## 运算
data = [1, -2, -1, 2]
tensor = torch.FloatTensor(data) # 转化为32bit 浮点数的tensor格式数据

print(
    '\n abs: ',
    '\n data: ', np.abs(data),
    '\n tensor: ', torch.abs(tensor),

    '\n sin: ',
    '\n data: ', np.sin(data),
    '\n tensor: ', torch.sin(tensor),

    '\n mean: ',
    '\n data: ', np.mean(data),
    '\n tensor: ', torch.mean(tensor)
)

# 矩阵运算

data = [[1, 2], [3, 4]]
tensor = torch.FloatTensor(data)
data = np.array(data)

print(
    '\n data:', np.matmul(data, data),
    '\n tensor:', torch.mm(tensor, tensor)
)

"""


data = [1, 2, 3, 4]
tensor = torch.FloatTensor(data)

# require_grad 代表是否要计算梯度
variable = Variable(tensor, requires_grad=True)

print(
    '\n tensor: ', tensor,
    '\n variable: ', variable,
)

# 计算 v ^ 2 均值
t_out = torch.mean(tensor * tensor)
v_out = torch.mean(variable * variable)
print(
    '\n t_ouy: ', t_out,
    '\n v_out: ', v_out,
)

# 误差反向传递
v_out.backward()

# 输出variable的梯度
print(
    '\n grad: ', variable.grad,
)

# analysis
# v_out = 1 / 4 * sum(v * v)
# d(v_out) = 1 / 4 * 2 * variable = 1 / 2 * variable

print(
    '\n variable: ', variable,
    '\n data: ', variable.data,
    '\n numpy data: ', variable.data.numpy(),
)