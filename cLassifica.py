"""
View more, visit my tutorial page: https://mofanpy.com/tutorials/
My Youtube Channel: https://www.youtube.com/user/MorvanZhou

Dependencies:
torch: 0.4
matplotlib
"""
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.autograd import Variable

# torch.manual_seed(1)    # reproducible

# make fake data
n_data = torch.ones(100, 2)

#print(n_data)
#返回一个tensor，包含从给定参数means,std的离散正态分布中抽取随机数
x0 = torch.normal(2*n_data, 1)      # class0 x data (tensor), shape=(100, 2)

print(
    '\n shape of x0: ', x0.size(),
    '\n means: ', 2 * n_data,
    '\n std: ', 1,
    '\n x0: ', x0
)

# 将该部分标记为0
y0 = torch.zeros(100)               # class0 y data (tensor), shape=(100, 1)

# 同理， 将该部分标记为1
x1 = torch.normal(-2*n_data, 1)     # class1 x data (tensor), shape=(100, 2)
y1 = torch.ones(100)                # class1 y data (tensor), shape=(100, 1)
print(
    '\n shape of x0: ', x0.size(),
    '\n means: ', -2 * n_data,
    '\n std: ', 1,
    '\n x0: ', x1
)

# 将两组点拼接为一组数据  为一个 200 * 2 的tensor
x = torch.cat((x0, x1), 0).type(torch.FloatTensor)  # shape (200, 2) FloatTensor = 32-bit floating

# 将y 也进行拼接作为 每个点的 标签
y = torch.cat((y0, y1), ).type(torch.LongTensor)    # shape (200,) LongTensor = 64-bit integer

# The code below is deprecated in Pytorch 0.4. Now, autograd directly supports tensors
x, y = Variable(x), Variable(y)

plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=y.data.numpy(), s=100, lw=0, cmap='RdYlGn')
plt.show()


# 搭建神经网络用以计算
# method1

# class Net(torch.nn.Module):
#     def __init__(self, n_feature, n_hidden, n_output):
#         super(Net, self).__init__()
#         self.hidden = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
#         self.out = torch.nn.Linear(n_hidden, n_output)   # output layer
#
#     def forward(self, x):
#         x = F.relu(self.hidden(x))      # activation function for hidden layer
#         x = self.out(x)
#         return x

class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net,self).__init__() # 继承部分初始化
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.out = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = F.relu(self.hidden(x)) # 对隐藏层神经元进行激活
        x = self.out(x)
        return x

## 初始化神经网络
# 初始化两个输入 两个输出
net1 = Net(n_feature=2, n_hidden=10, n_output=2)     # define the network
print(net1)  # net architecture

# method2 快速搭建法

net2 = torch.nn.Sequential( # 借助Sequential 方法
    torch.nn.Linear(2, 10), #设置隐藏层参数 2层输入 10 层输出
    
    torch.nn.ReLU(), # 激活函数进行激活 激活函数是一个类

    torch.nn.Linear(10, 2), # 输出层 设置
)

print(net2)

# optimizer = torch.optim.SGD(net.parameters(), lr=0.02)
# loss_func = torch.nn.CrossEntropyLoss()  # the target label is NOT an one-hotted
# # 输出为[0, 1] 则将标签贴到第二类认为当前输入为第二类
# # 输出为[1, 0] 则将标签贴到第一类认为当前输入为第一类
# # 损失函数采取 CrossEntropy
# # [0, 0, 1]
# # [0.2, 0.3, 0.5]
# # 将输出与标准输出对比得到 误差值 作为损失函数
#
# plt.ion()   # something about plotting
#
# # 开始训练过程
# for t in range(100):
#     out = net(x)                 # input x and predict based on x
#     loss = loss_func(out, y)     # must be (1. nn output, 2. target), the target label is NOT one-hotted
#
#     optimizer.zero_grad()   # clear gradients for next train
#     loss.backward()         # backpropagation, compute gradients
#     optimizer.step()        # apply gradients
#
#     if t % 2 == 0:
#         # plot and show learning process
#         plt.cla()
#         # [1] 代表返回最大值的索引 [0]代表返回最大值本身
#         prediction = torch.max(out, 1)[1]
#
#         print(
#             '\n prediction: ', prediction,
#         )
#         # 按照下标
#         pred_y = prediction.data.numpy()
#         target_y = y.data.numpy()
#         plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn')
#         accuracy = float((pred_y == target_y).astype(int).sum()) / float(target_y.size)
#         plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 20, 'color':  'red'})
#         plt.pause(0.1)
#
# plt.ioff()
# plt.show()
