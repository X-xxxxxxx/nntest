import matplotlib.pyplot as plt
import numpy as np

# x = np.linspace(-1, 1, 50)
# y1 = 2 * x + 1
# y2 = x ** 2
# plt.plot(x, y)
# plt.show()


# figure 使用
# plt.figure()
# plt.plot(x, y1)

##plt.figure(num='3', figsize=(8, 5))
# plt.figure()
# plt.plot(x, y2)
# plt.plot(x, y1, color='red', lw=1.0, linestyle='--')
#
# # 取值范围
# plt.xlim((-1, 2))
# plt.ylim((-2, 3))
#
# # x, y label
# plt.xlabel("i am x")
# plt.ylabel("i am y")
#
# # 更换角标
# new_ticks = np.linspace(-1, 2, 5)
# print(new_ticks)
# plt.xticks(new_ticks)
# plt.yticks([-2, -1.8, -1, 1.22, 3, ], [r'$really\ bad$', r'$bad\ \alpha$', r'$normal$', r'$good$', r'$really\ good$'])
#
# # 移动坐标轴
# # gca = ‘get current axis’
# ax = plt.gca()
# ax.spines['right'].set_color('none')
# ax.spines['top'].set_color('none')
# ax.xaxis.set_ticks_position('bottom')
# ax.yaxis.set_ticks_position('left')
#
# ax.spines['bottom'].set_position(('data', 0))   # outward, axes
# ax.spines['left'].set_position(('data', 0))
#
#
#
# # legend 设置
#
# l1, = plt.plot(x, y2, label='up')   # 注意逗号
# l2, = plt.plot(x, y1, color='red', lw=1.0, linestyle='--', label='down')
# plt.legend(handles=[l1, l2], labels=['uu', 'dd'], loc='best', )  # loc = ['best', upper, lower]

# annotation 标注
# 为图形中需要特别强调的点进行标注

# x = np.linspace(-3, 3, 50)
# #y = 2 * x + 1
# y = 0.01 * x
#
# plt.figure()
# plt.plot(x, y, lw=10)
# plt.xlim(-2, 2)
# ax = plt.gca()
# ax.spines['right'].set_color('none')
# ax.spines['top'].set_color('none')
#
# ax.xaxis.set_ticks_position('bottom')
# ax.yaxis.set_ticks_position('left')
#
# ax.spines['bottom'].set_position(('data', 0))
# ax.spines['left'].set_position(('data', 0))

# # 标注一个感兴趣点
# x0 = 1
# y0 = 2 * x0 + 1
#
# plt.scatter(x0, y0,s=50, color='b')
# plt.plot([x0, x0], [y0, 0], 'k--', lw=2.5)
#
#
# # method 1
# #####
# #plt.annotate(r'$2x + 1= %s$' % y0, xy=(x0, y0), xycoords='data', xytext=(+30.-30), testcoords='offset point',) # 正则表达设置文字
# plt.annotate(r'$2x + 1=%s$' % y0, xy=(x0, y0), xycoords='data', xytext=(+30, -30),
#              textcoords='offset points', fontsize=16,
#              arrowprops=dict(arrowstyle='->', connectionstyle="arc3,rad=.2"))
#
# # method 2
#
# plt.text(-3.7, 3, r'$this\ is\ some\ test.'
#                   r'\mu\ \sigma_i\ \alpha_t$',
#          fontdict={'size':16, 'color':'r'})


### 设置能见度
#
# for label in ax.get_xticklabels() + ax.get_yticklabels():
#     label.set_fontsize(12)  # 设置大小
#     label.set_bbox(dict(facecolor='white', edgecolor='red', alpha=0.3))   # alpha 透明度

## 散点图
# n = 1024
# X = np.random.normal(0, 1, n)   # 生成n个均值为0方差为1的随机数
# Y = np.random.normal(0, 1, n)
#
# # 颜色
# T = np.arctan2(Y, X)
#
# plt.scatter(X, Y, s=75, c=T, alpha=0.5)
#
# # plt.scatter(np.arange(5), np.arange(5))
# # plt.xlim(-1.5, 1.5)
# # plt.ylim(-1.5, 1.5)
#
# # plt.xticks(())
# # plt.yticks(())

# ## 条形图
#
# # data
#
# n = 12
# X = np.arange(n)
#
# Y1 = (1 - X / float(n)) * np.random.uniform(0.5, 1.0, n)
# Y2 = (1 - X / float(n)) * np.random.uniform(0.5, 1.0, n)
#
# plt.bar(X, +Y1, facecolor='#9999ff', edgecolor='white')
# plt.bar(X, -Y2, facecolor='#ff9999', edgecolor='white')
#
#
# # 加上标签
#
# for x, y in zip(X,Y1):
#     # ha horizontal alignment 横向对齐
#     plt.text(x + 0.4, y + 0.05, '%.2f' % y, ha='right', va='bottom')
#
# for x, y in zip(X,Y1):
#     # ha horizontal alignment 横向对齐
#     plt.text(x + 0.4, -y - 0.05, '%.2f' % y, ha='right', va='top')


#
# ## 等高线图
# def func(x, y):
#     return (1 - x / 2 + x**5 + y**3) * np.exp(-x**2 -y**2)
#
# n = 256
# x = np.linspace(-3, 3, n)
# y = np.linspace(-3, 3, n)
#
# # 网格
# X, Y = np.meshgrid(x, y)
#
# print(X, Y)
# # plt.xticks()
# # plt.yticks()
#
# plt.contourf(X, Y, func(X, Y), 10, alpha=0.75, cmap=plt.cm.cool)
#
# # 画等高线的线条
# C = plt.contour(X, Y, func(X, Y), 10, colors='black', linewidths=5)  # 10 代表分为多少块 0 代表两块 记得加上s
# #  加上标签
# plt.clabel(C, inline=True, fontsize=10)


# # 图像绘制
# # image data
# a = np.array([0.313660827978, 0.365348418405, 0.423733120134,
#               0.365348418405, 0.439599930621, 0.525083754405,
#               0.423733120134, 0.525083754405, 0.651536351379]).reshape(3,3)
#
#
# plt.imshow(a, interpolation='nearest', cmap='bone', origin='lower')
#
# ##  标注
# plt.colorbar(shrink=0.9)



# 3D 图像绘制

# 额外模块
# from mpl_toolkits.mplot3d import Axes3D
# #
# fig = plt.figure()
# ax = Axes3D(fig)
# fig.add_axes(ax)    # 添加
# X = np.arange(-4, 4, 0.25)
# Y = np.arange(-4, 4, 0.25)
# X, Y = np.meshgrid(X, Y)
# R = np.sqrt(X ** 2 + Y ** 2)
# # height value
# Z = np.sin(R)
#
# ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'))    # stride 为跨度
#
# ax.contourf(X, Y, Z, zdir='x', offset=-4, cmap='rainbow') # zdir 代表从哪边进行投影
#
# ax.set_zlim(-2, 2)


# # 分块显示 subplot
# plt.figure()
# # plt.subplot(2, 2, 1)
# # plt.plot([0, 1], [0, 1])
# # plt.subplot(2, 2, 2)
# # plt.plot([0, 1], [0, 2])
# # plt.subplot(2, 2, 3)
# # plt.plot([0, 1], [0, 3])
# # plt.subplot(2, 2, 4)
# # plt.plot([0, 1], [0, 4])
#
# plt.subplot(2, 1, 1)
# plt.plot([0, 1], [0, 1])
# plt.subplot(2, 3, 4)
# plt.plot([0, 1], [0, 2])
# plt.subplot(2, 3, 5)
# plt.plot([0, 1], [0, 3])
# plt.subplot(2, 3, 6)
# plt.plot([0, 1], [0, 4])



# #  分格显示
#
# from matplotlib.gridspec import GridSpec
#
# plt.figure()
# ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=3, rowspan=1)  # (3, 3) 三行散列 (0, 0) 从0 0 开始plot
# ax1.plot([1, 2], [1, 2])
# ax1.set_title('ax1_title')
#
#
# ax2 = plt.subplot2grid((3, 3), (1, 0), colspan=2, rowspan=1)
# ax3 = plt.subplot2grid((3, 3), (1, 2), colspan=1, rowspan=2)
# ax4 = plt.subplot2grid((3, 3), (2, 0),)
# ax5 = plt.subplot2grid((3, 3), (2, 1),)
#
#
# plt.figure()
#
# gs = GridSpec.Gridspec(3, 3)
# plt.show()
#



### 图中图
# fig = plt.figure()
#
# x = [1, 2, 3, 4, 5, 6, 7]
# y = [1, 3, 4, 2, 5, 8, 6]
# left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
#
# ax1 = fig.add_axes([left, bottom, width, height])
#
# ax1.plot(x, y, 'r')
# ax1.set_xlabel('x')
# ax1.set_ylabel('y')
# ax1.set_title('title')
#
# left, bottom, width, height = 0.2, 0.6, 0.25, 0.25
#
# ax2 = fig.add_axes([left, bottom, width, height])
#
# ax2.plot(x, y, 'r')
# ax2.set_xlabel('x')
# ax2.set_ylabel('y')
# ax2.set_title('title inside 1')
#
#
# plt.axes([0.6, 0.2, 0.25, 0.25])
# plt.plot(y[::-1], x, 'g')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('inside 2')
# plt.show()

## 主次坐标轴
#
#
# x = np.arange(0, 10, 0.1)
#
# y1 = 0.05 * x ** 2
# y2 = -1 * y1
#
# fig, ax1 = plt.subplots()
#
# ax2 = ax1.twinx()
#
#
# ax1.plot(x, y1, 'g-')
# ax2.plot(x, y2, 'b-')
# ax1.set_xlabel('X data')
# ax1.set_ylabel('Y1 data', color='g')
# ax2.set_ylabel('Y2 data', color='b')
#
# plt.show()



# 动画
from matplotlib import animation
fig, ax = plt.subplots()
x = np.arange(0, 2 * np.pi, 0.01)
line, = ax.plot(x, np.sin(x))

def animat(i):
    line.set_ydata(np.sin(x + i / 10))
    return line,

def init():
    line.set_ydata(np.sin(x))
    return line,

ani = animation.FuncAnimation(fig=fig,
                              func=animat,
                              frames=100,
                              init_func=init,
                              interval=20,  # 20ms 更新一次
                              blit=True,    # 是否更新整张图形的点 为False 否则只能更新变化点
                              )

plt.show()