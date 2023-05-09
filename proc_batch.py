import torch
import torch.utils.data as Data
import matplotlib.pyplot as plt


# 数据量非常大，需要分批次进行训练

BATCH_SIZE = 8  # 数据集中每批的个数

# fake data
x = torch.linspace(1, 10, 10)
y = torch.linspace(10, 1, 10)

# plt.figure(1, figsize=(8, 6))
# plt.scatter(x.data.numpy(), y.data.numpy())
# plt.show()

# 关于数据预处理
torch_dataset = Data.TensorDataset(x, y)  # x为训练数据 y为目标数据 用以计算实际误差
loader = Data.DataLoader(
    dataset=torch_dataset,  # 数据集
    batch_size=BATCH_SIZE,  # 设置批训练尺寸
    shuffle=True,   # 是否将数据打乱训练
    num_workers=2,  # 使用双线程进行数据提取
)


def show_batch():
    for epoch in range(3):  # 训练轮次
        for step, (batch_x, batch_y) in enumerate(loader):
            # training ...
            print('Epoch: ', epoch, '| Step: ', step, '| batch x: ',
                  batch_x.numpy(), '| batch y: ', batch_y.numpy())



if __name__ == '__main__':
    show_batch()