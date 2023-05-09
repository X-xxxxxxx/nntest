import torch
import torch.utils.data as Data
import matplotlib.pyplot as plt
import torch.nn.functional as F


LR = 0.01   # 学习率
BATCH_SIZE = 5  # 批学习数量
EPOCH = 10  #学习轮数

x = torch.unsqueeze(torch.linspace(-1, 1, 1000), dim=1)
y = x.pow(2) + 0.1*torch.normal(torch.zeros(*x.size()))


plt.scatter(x.data.numpy(), y.data.numpy())
plt.show()

# 设置数据集

torch_dataset = Data.TensorDataset(x, y)
loader = Data.DataLoader(
    dataset=torch_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2
)

# 初始化网络

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(1, 20)
        self.predict = torch.nn.Linear(20, 1)

    def forward(self, x):
        x = F.relu(self.hidden(x))  # 激活隐藏层
        x = self.predict(x)

        return x


if __name__ == '__main__':
    # s定义四个网络分别使用不同的优化器
    net_SGD         = Net()
    net_Momentum    = Net()
    net_RMSprop     = Net()
    net_Adam        = Net()

    # 将网络放入一个列表 方便训练
    nets = [net_SGD, net_Momentum, net_RMSprop, net_Adam]

    # 对四个网络用不同的Optimizer
    opt_SGD         = torch.optim.SGD(net_SGD.parameters(), lr=LR)
    opt_Momentum    = torch.optim.SGD(net_Momentum.parameters(), lr=LR, momentum=0.8)
    opt_RMSprop     = torch.optim.RMSprop(net_RMSprop.parameters(), lr=LR, alpha=0.9)
    opt_Adam        = torch.optim.Adam(net_Adam.parameters(), lr=LR, betas=(0.9, 0.99))

    optimizers = [opt_SGD, opt_Momentum, opt_RMSprop, opt_Adam]



    # 定义损失函数
    loss_func = torch.nn.MSELoss()
    losses_his = [[], [], [], []]

    # training...

    for epoch in range(EPOCH):
        print('Epoch: ', epoch)

        for step, (b_x, b_y) in enumerate(loader):
            for net, opt, l_his in zip(nets, optimizers, losses_his):
                output = net(b_x)   # 拿到输出
                loss = loss_func(output, b_y)   # 计算误差
                opt.zero_grad()    #clear gradients

                loss.backward() # 反向传播
                opt.step()  #接受梯度
                l_his.append(loss.data.numpy()) # 记录损失

    labels = ['SGD', 'Momentum', 'RMSprop', 'Adam']

    for i, l_his in enumerate(losses_his):
        plt.plot(l_his, label=labels[i])
    plt.legend(loc='best')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    # plt.ylim(0, 0.2)
    # plt.xlim(0, 400)

    plt.show()
