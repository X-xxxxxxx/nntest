import numpy as np

array = np.array([[1, 2, 3],
                [2, 3, 4]])


print(array)


print('number of dim: ', array.ndim)
print('shape: ', array.shape)
print('size:', array.size)
print('data', array.data)


a = np.array([10, 20, 30, 40])
b = np.arange(4)

print(a, b)
print(a * b)

c = 10 * np.sin(a)  # 输入的为弧度

print(c)
print(b)
print(b < 3)    #返回一个bool 类型列表

a = np.array([[0, 1], [1, 2]])
b = np.arange(4).reshape(2, 2)

print(a)
print(b)
print(a * b)
print(np.dot(a, b))
print(a.dot(b))


a = np.random.random((2, 4))    # (0, 1) random
print(a)

print(np.sum(a, axis=0))
print(np.min(a, axis=0))    # 列
print(np.max(a, axis=1))    # 行


a = np.arange(0, 12).reshape(3, 4)

print(a)
print(np.argmin(a))  # 索引
print(np.argmax(a))  # 索引


# 求平均值
print(np.mean(a))
print(a.mean())
print(np.average(a))

# 求中位数
print(np.median(a))

# 求前缀和
print(a)
print(np.cumsum(a))

# 差分
print(a)
print(np.diff(a))

# 输出非0值的索引
print(np.nonzero(a))

# 按行排序
a = np.arange(14, 2, -1).reshape(3, 4)
print(a)
print(np.sort(a))

# 矩阵转置
print(np.transpose(a))
print(a.T)


# 矩阵切片
print(np.clip(a, 5, 9)) # 大于最大值的变为最大值 小于最小值的变为最小值 中间的不变

print(np.mean(a, axis=0))


a = np.arange(3, 15)
print(a)
print(a[3])

a = np.arange(3, 15).reshape(3, 4)
print(a)
print(a[0][1])
print(a[0, 1])
print(a[2, :])  # : 号代表所有数字

for row in a:
    print(row)

for col in a.T:
    print(col)

# 读取每一个值
print(a.flatten())
for item in a.flat:
    print(item)