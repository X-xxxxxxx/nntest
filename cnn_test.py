# import os
#
# # third-party library
# import torch
# import torch.nn as nn
# import torch.utils.data as Data
# import torchvision
# import torch.autograd.variable as Variable
#
# import matplotlib.pyplot as plt
#
# # torch.manual_seed(1)    # reproducible
#
# # Hyper Parameters
# EPOCH = 1               # train the training data n times, to save time, we just train 1 epoch
# BATCH_SIZE = 50
# LR = 0.001              # learning rate
# DOWNLOAD_MNIST = False
#
#
# # Mnist digits dataset
# if not(os.path.exists('./mnist/')) or not os.listdir('./mnist/'): # 如果没有生成了该文件或者还没有该文件 则是否下载置为true
#     # not mnist dir or mnist is empyt dir
#     DOWNLOAD_MNIST = True
#
# train_data = torchvision.datasets.MNIST(
#     root='./mnist/',                                # 保存路径
#     train=True,                                     # this is training data 为true则代表下载训练数据
#     transform=torchvision.transforms.ToTensor(),    # Converts a PIL.Image or numpy.ndarray to
#                                                     # torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0] # 将像素值进行压缩
#     download=DOWNLOAD_MNIST,
# )
#
#
# # print(train_data.train_data.size())                 # (60000, 28, 28)
# # print(train_data.train_labels.size())               # (60000)
# # plt.imshow(train_data.train_data[0].numpy(), cmap='gray')
# # plt.title('%i' % train_data.train_labels[0])
# # plt.show()
#
#
# # 数据准备
# # train_loador = Data.DataLoader(
# #     dataset=train_data,
# #     batch_size=BATCH_SIZE,
# #     shuffle=True,
# #     num_workers=1,
# # )
# # # 测试集 准备
# # test_data = torchvision.datasets.MNIST(
# #     root='./mnist/',
# #     train=False,
# # )
# # test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)[:2000]/255.   # shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
# # test_y = test_data.test_labels[:2000]
# # print(test_x.size())
#
#
# # Data Loader for easy mini-batch return in training, the image batch shape will be (50, 1, 28, 28)
# train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
#
# # pick 2000 samples to speed up testing
# test_data = torchvision.datasets.MNIST(root='./mnist/', train=False)
# test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)[:2000]/255.   # shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
# test_y = test_data.test_labels[:2000]
#
#
# # class CNN(nn.Module):
# #     def __init__(self):
# #         super(CNN, self).__init__()
# #         # 卷积层设置
# #         self.conv1 = nn.Sequential(
# #             nn.Conv2d(
# #                 in_channels=1,  # 图像的层数
# #                 kernel_size=5,  # 卷积核的大小
# #                 out_channels=16,    # 使用了16个卷积核因此一个图像经过该卷积层可有16个输出
# #                 stride=1,   #卷积核每次移动一个
# #                 padding=2,  # 当图像池逊
# #             ), # 16 * 28 * 28
# #             nn.ReLU(), # 16 * 28 * 28
# #             nn.MaxPool2d(
# #                 kernel_size=2,  # 类似于卷积核的尺寸
# #             ), # 16 * 14 * 14
# #         )
# #         self.conv2 = nn.Sequential( # 16 * 14 * 14
# #                 nn.Conv2d(16, 32, 5, 1, 2), # 32 * 14 * 14
# #                 nn.ReLU(),
# #                 nn.MaxPool2d(2), # 32 * 7 * 7
# #             )
# #         self.out = nn.Linear(32 * 7 * 7, 10)    # 10 个数字  10个输出
# #
# #     def forward(self, x):
# #         x = self.conv1(x)
# #         x = self.conv2(x) # shape = (batch, 32, 7, 7)
# #         x = x.view(x.size(0), -1)   # shape = (batch, 32 * 7 * 7)
# #         output = self.out(x)
# #
# #         return output
#
# class CNN(nn.Module):
#     def __init__(self):
#         super(CNN, self).__init__()
#         self.conv1 = nn.Sequential(         # input shape (1, 28, 28)
#             nn.Conv2d(
#                 in_channels=1,              # input height
#                 out_channels=16,            # n_filters
#                 kernel_size=5,              # filter size
#                 stride=1,                   # filter movement/step
#                 padding=2,                  # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
#             ),                              # output shape (16, 28, 28)
#             nn.ReLU(),                      # activation
#             nn.MaxPool2d(kernel_size=2),    # choose max value in 2x2 area, output shape (16, 14, 14)
#         )
#         self.conv2 = nn.Sequential(         # input shape (16, 14, 14)
#             nn.Conv2d(16, 32, 5, 1, 2),     # output shape (32, 14, 14)
#             nn.ReLU(),                      # activation
#             nn.MaxPool2d(2),                # output shape (32, 7, 7)
#         )
#         self.out = nn.Linear(32 * 7 * 7, 10)   # fully connected layer, output 10 classes
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = x.view(x.size(0), -1)           # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
#         output = self.out(x)
#         return output, x    # return x for visualization
#
# cnn = CNN()
# print(cnn)
#
#
# # 优化器设置
# optimize = torch.optim.Adam(cnn.parameters(), lr=LR)
# loss_func = nn.CrossEntropyLoss()
#
#
# for epoch in range(EPOCH):
#
#     for step, (b_x, b_y) in enumerate(train_loader):
#
#         output = cnn(b_x)[0]
#
#         loss = loss_func(output, b_y)
#
#         optimize.zero_grad()    # 梯度清零
#         loss.backward() # 反向传播
#         optimize.step()
#         # 每50步 输出一次
#         if step % 50 == 0:
#             test_output, _ = cnn(test_x)
#
#             pred_y = torch.max(test_output, 1)[1].data.numpy()
#             accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
#             print('Epoch: ', epoch, '| train loss: %.4f' % loss.data[0], 'test accuracy: ', accuracy)
#
#
#
# test_output = cnn(test_x[:10])
# pred_y = torch.max(test_output, 1)[1].data.numpy.squeeze()
# print(pred_y, 'prediction number')
# print(test_y[:10].numpy(), 'real number')

import os
import torch
import torch.nn as nn
import torch.utils.data as Data # 数据集
import torchvision
import matplotlib.pyplot as plt


# 设置参数
EPOCH = 1
BATCH_SIZE = 50
LR = 0.01
DOWNLOAD_MINST = False

if not(os.path.exists('./mnist/')) or not os.listdir('./mnist/'): # 如果没有生成了该文件或者还没有该文件 则是否下载置为true
    # not mnist dir or mnist is empyt dir
    DOWNLOAD_MINST = True

# 数据集下载

train_data = torchvision.datasets.MNIST(
    root='./mnist/',
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=DOWNLOAD_MINST
)

# 数据加载
train_loader = Data.DataLoader(
    dataset=train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

# 测试集准备
test_data = torchvision.datasets.MNIST(
    root='./mnist',
    train=False,
)

test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)[:2000]/255 # shape from (2000, 28, 28) to (2000, 1, 28, 28)

# print(test_data.test_data.shape)
# print(test_x.shape)
test_y = test_data.test_labels[:2000]


# cnn 网络搭建

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 卷积层1
        self.conv1 = nn.Sequential(
            # 卷积核
            nn.Conv2d(
                in_channels=1,
                out_channels=16, # 使用16个卷积核
                kernel_size=5, # 卷积核大小为5 * 5
                stride=1, # 每次移动一步
                padding=2,
            ),
            nn.ReLU(), # 激活
            nn.MaxPool2d(kernel_size=2), # 池化
        )
        # 卷积层2
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # 全连接层
        self.out = nn.Linear(32 * 7 * 7, 10)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1) # shape from batch_size , 32, 7, 7 to batch_size, 32 * 7 * 7

        output = self.out(x)

        return output, x

# 初始化cnn网络
cnn = CNN()
print(cnn)

# 设置优化器和损失函数

optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()

# 训练以及测试

try:
    from sklearn.manifold import TSNE;
    HAS_SK = True
except:
    HAS_SK = False;
    print('Please install sklearn for layer visualization')


print(HAS_SK)

from matplotlib import cm

def plot_with_labels(lowDWeights, labels):
    plt.cla()
    X, Y = lowDWeights[:, 0], lowDWeights[:, 1]
    for x, y, s in zip(X, Y, labels):
        c = cm.rainbow(int(255 * s / 9)); plt.text(x, y, s, backgroundcolor=c, fontsize=9)
    plt.xlim(X.min(), X.max()); plt.ylim(Y.min(), Y.max()); plt.title('Visualize last layer'); plt.show(); plt.pause(0.01)


plt.ion()
for epoch in range(EPOCH):

    for step, (d_x, d_y) in enumerate(train_loader):
        output = cnn(d_x)[0]
        loss = loss_func(output, d_y)
        optimizer.zero_grad() # 梯度清零
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            # 测试集测试
            test_output, last_layer = cnn(test_x)
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            accuracy = (pred_y == test_y).sum().item() / float(test_y.size(0))
            # sum()相加后为张量类型的int 值  item() 将其转化为python类型的数字值
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data, '| test accuracy: %.2f' % accuracy)

            if HAS_SK:
                # Visualization of trained flatten layer (T-SNE)
                tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
                plot_only = 500
                low_dim_embs = tsne.fit_transform(last_layer.data.numpy()[:plot_only, :])
                labels = test_y.numpy()[:plot_only]
                plot_with_labels(low_dim_embs, labels)


plt.ioff()

test_output, _ = cnn(test_x[:10])
pred_y = torch.max(test_output, 1)[1].data.squeeze()
print(pred_y, 'prediction number')
print(test_y[:10].numpy(), 'real number')