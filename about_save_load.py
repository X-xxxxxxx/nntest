import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.autograd import Variable


# torch.manual_seed(1)    # reproducible

# fake data
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  # x data (tensor), shape=(100, 1)
y = x.pow(2) + 0.2*torch.rand(x.size())                 # noisy y data (tensor), shape=(100, 1)

x, y = Variable(x, requires_grad=False), Variable(y, requires_grad=False)


def save():
    # 快速搭建一个神经网络
    net1 = torch.nn.Sequential(
        torch.nn.Linear(1, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1),
    )

    optimizer = torch.optim.SGD(net1.parameters(), lr=0.5)
    loss_func = torch.nn.MSELoss()

    for i in range(1, 100):
        prediction = net1(x)
        loss = loss_func(prediction, y)
        optimizer.zero_grad() # 梯度清零
        loss.backward()
        optimizer.step()


    # 可视化网络用以对比
    # plot result
    plt.figure(1, figsize=(10, 3)) #设置图片尺寸
    plt.subplot(131)
    plt.title('Net1')
    plt.scatter(x.data.numpy(), y.data.numpy()) # 完成散点图绘制
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5) # 完成训练结果图进行对比

    # 保存模型
    torch.save(net1, 'net.pkl') # entire net
    torch.save(net1.state_dict(), 'net_para.pkl') # parameters


def restore_net():
    # 导入整个网络模型

    net2 = torch.load('net.pkl')
    prediction = net2(x) # 拿到训练结果
    plt.subplot(132)
    plt.title('Net2')
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', linewidth=5)
    #plt.show()
def restore_params():
    # 导入网络参数
    # 先搭建一个相同结构的网络
    net3 = torch.nn.Sequential(
        torch.nn.Linear(1, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1),
    )

    net3.load_state_dict(torch.load('net_para.pkl'))
    prediction = net3(x)
    plt.subplot(133)
    plt.title('Net3')
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', linewidth=5)
    plt.show()

save()
restore_net()
restore_params()

## 可视化part


