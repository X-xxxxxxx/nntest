import pandas as pd
import numpy as np

# create 列表
s = pd.Series([1, 3, 6, np.nan, 44, 1])

print(s)

dates = pd.date_range('20230507', periods=6)

print(dates)

# 生成表数据
df = pd.DataFrame(np.random.random(), index=dates, columns=['a', 'b', 'c', 'd'])
print(df)


df = pd.DataFrame(np.arange(12).reshape((3, 4)))

print(df)

# 以字典形式进行二维数据生成
df2 = pd.DataFrame({'A' : 1.,
                    'B' : pd.Timestamp('20130102'),
                    'C' : pd.Series(1,index=list(range(4)),dtype='float32'),
                    'D' : np.array([3] * 4,dtype='int32'),
                    'E' : pd.Categorical(["test","train","test","train"]),
                    'F' : 'foo'})
# print(df2)
# print(df2.dtypes)   #类型
# print(df2.index)    # 索引
# print(df2.columns)  # 列名
# print(df2.values)   # 值
# print(df2.describe())   # 属性描述
# print(df2.T)    #转置
#
# print(df2.sort_index(axis=1, ascending=False))  # ascending == False 则以倒序排序
# print(df2.sort_index(axis=0, ascending=False))
#
#
# print(df2.sort_values(by='E'))  # 对第e列进行排序
#
dates = pd.date_range('20130101', periods=6)
df = pd.DataFrame(np.arange(24).reshape((6, 4)), index=dates, columns=['A', 'B', 'C', 'D'])
# print(df)
# print(df['A'], df.A)

# SELECT BY LABEL: LOC
# 使用loc 进行选择
print(df)
print(df.loc['20130102'])

# print(df.loc[:,['A', 'B']])
# print(df.loc['20130102', ['A', 'B']])
#
#
# # select by position: iloc
#
# print(df.iloc[[1, 3, 5], 1:3])
#
# # boolean indexing 按条件选择
#
# print(df[df.A > 8])
#
# dates = pd.date_range('20130101', periods=6)
# df = pd.DataFrame(np.arange(24).reshape((6, 4)), index=dates, columns=['A', 'B', 'C', 'D'])

# df.iloc[2, 2] = 1111
# df.loc['20130101', 'B'] = 2222
#
# # 批量修改
# df.A[df.A>4] = 0
#
# # 可直接新生成列
# df['F'] = np.nan
#
# df['E'] = pd.Series([1, 2, 3, 4, 5, 6], index=pd.date_range('20130101',periods=6))
# print(df)


# # 处理丢失数据
#
# # 假设丢失数据
# df.iloc[0, 1] = np.nan
# df.iloc[1, 2] = np.nan
#
# # 处理方法
# print(df.dropna(axis=0, how='any'))  # 是要有nan 就 按行丢掉
# print(df.dropna(axis=0, how='all'))  # 该行所有为nan才按行丢掉
#
# print(df.fillna(value=0))   # 将为nan 的数据以value填充上
#
# # 判断是否有缺失数据
# print(df.isnull())
# print(np.any(df.isnull() == True))



## pandas 导入 导出


#
# data = pd.read_csv(r"C:\Users\X_xx\Desktop\student.csv", encoding='gbk')
#
# data.to_pickle(r'student.pickle')
# print(data)
#
#
#
# # dataframe 合并
#
# # concatenating
#
#
#
# # data 相同的columns
# # 上下合并
# df1 = pd.DataFrame(np.ones((3, 4)) * 0, columns=['a', 'b', 'c', 'd'])
# df2 = pd.DataFrame(np.ones((3, 4)) * 1, columns=['a', 'b', 'c', 'd'])
# df3 = pd.DataFrame(np.ones((3, 4)) * 2, columns=['a', 'b', 'c', 'd'])
#
#
# res = pd.concat([df1, df2, df3], axis=0)    # 合并列 索引并未变化
# print(res)
# res = pd.concat([df1, df2, df3], axis=0, ignore_index=True)     # 忽略索引
# print(res)
#
# #join.['inner', 'outer']
#
# # columns 和 index 又不同
# df1 = pd.DataFrame(np.ones((3, 4)) * 0, columns=['a', 'b', 'c', 'd'], index=[1, 2, 3])
# df2 = pd.DataFrame(np.ones((3, 4)) * 1, columns=['b', 'c', 'd', 'e'], index=[2, 3, 4])
#
# df3 = pd.DataFrame(np.ones((3, 4)) * 0, columns=['a', 'b', 'c', 'd'], index=[1, 2, 3])
# df4 = pd.DataFrame(np.ones((3, 4)) * 0, columns=['a', 'b', 'c', 'd'], index=[1, 2, 3])
#
# print(df1)
# print(df2)
# res = pd.concat([df1, df2], join='outer')   # 默认为outer 将没有数据的地方赋值 nan
# print(res)
# res = pd.concat([df1, df2], join='inner', ignore_index=True)   # 合并时候只考虑两者都有的东西
# print(res)
#
# # res = pd.concat([df1, df2], axis=1, join_axes=[df1.index])    # pandas 1.0.0 之后已经弃用 join_axes
# # print(res)
#
# s1 = pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])
#
# res = df1._append(s1, ignore_index=True)
# print(res)
#
# res = df1._append([df3, df4], )
# print(res)
#
#
# ## merge
#
# left = pd.DataFrame(
#                     {'key': ['K0', 'K1', 'K2', 'K3'],
#                     'A': ['A0', 'A1', 'A2', 'A3'],
#                     'B': ['B0', 'B1', 'B2', 'B3']}
# )
# right = pd.DataFrame(
#                     {'key': ['K0', 'K1', 'K2', 'K3'],
#                     'c': ['C0', 'C1', 'C2', 'C3'],
#                     'D': ['D0', 'D1', 'D2', 'D3']}
# )
#
# print(left.dtypes)
# print(right)
#
# res = pd.merge(left, right, on='key')   # 基于 key columns 合并
# print(res)
#
# # consider two keys
# left = pd.DataFrame({'key1': ['K0', 'K0', 'K1', 'K2'],
#                              'key2': ['K0', 'K1', 'K0', 'K1'],
#                              'A': ['A0', 'A1', 'A2', 'A3'],
#                              'B': ['B0', 'B1', 'B2', 'B3']})
# right = pd.DataFrame({'key1': ['K0', 'K1', 'K1', 'K2'],
#                               'key2': ['K0', 'K0', 'K0', 'K0'],
#                               'C': ['C0', 'C1', 'C2', 'C3'],
#                               'D': ['D0', 'D1', 'D2', 'D3']})
#
# # print(left)
# # print(right)
# #
# # # how = ['left', 'right', 'innner', 'outer']
# # # res = pd.merge(left, right, on=['key1', 'key2'], how='inner')    # 默认合并 key1 和 key2 均存在的数据
# # # print(res)
# # #
# # # res = pd.merge(left, right, on=['key1', 'key2'], how='outer')
# # # print(res)
# #
# # res = pd.merge(left, right, on=['key1', 'key2'], how='right')   # 以right key为对照
# # print(res)
#
#
# # indicator
# df1 = pd.DataFrame({'col1':[0,1], 'col_left':['a','b']})
# df2 = pd.DataFrame({'col1':[1,2,2],'col_right':[2,2,2]})
# print(df1)
# print(df2)
#
#
#
# # res = pd.merge(df1, df2, on='col1', how='outer')
# # print(res)
#
# # res = pd.merge(df1, df2, on='col1', how='outer', indicator=True)    # 指出合并的是哪一个dataframe的数据
# # print(res)
# # res = pd.merge(df1, df2, on='col1', how='outer', indicator='indicator_columns')    # 指出合并的是哪一个dataframe的数据
# # print(res)
# #
#
# #
# # # merged by index
# # left = pd.DataFrame(
# #                     {'A': ['A0', 'A1', 'A2'],
# #                     'B': ['B0', 'B1', 'B2']},
# #                                   index=['K0', 'K1', 'K2'])
# # right = pd.DataFrame(
# #                     {'C': ['C0', 'C2', 'C3'],
# #                     'D': ['D0', 'D2', 'D3']},
# #                                       index=['K0', 'K2', 'K3'])
# #
# # print(left)
# # print(right)
# # res = pd.merge(left, right, left_index=True, right_index=True, how='outer')    # 由考虑columns 变为index
# # print(res)
#
# # # handle overlapping
# # boys = pd.DataFrame({'k': ['K0', 'K1', 'K2'], 'age': [1, 2, 3]})
# # girls = pd.DataFrame({'k': ['K0', 'K0', 'K3'], 'age': [4, 5, 6]})
# #
# # print(boys)
# # print(girls)
# #
# # res = pd.merge(boys, girls, on='k', suffixes=['_boys', '_girls'], how='inner')
# # print(res)
#
# # pandas 数据可视化
#
# import matplotlib.pyplot as plt
#
# # plot data
# # Series
# data = pd.Series(np.random.randn(1000), index=np.arange(1000))
#
# # print(data)
# # # 累加
# # data = data.cumsum()
# # print(data)
# # data.plot()
# #
# # plt.show()
#
# # data frame
#
# data = pd.DataFrame(np.random.randn(1000, 4), index=np.arange(1000), columns=list("ABCD"))
#
# # 输出头部数据默认为前5个
# print(data.head(5))
# #data = data.cumsum()
#
#
# # data.plot() # 很多参数
# # plt.show()
#
# # plot methods:
# # 'bar' 条形图, 'his', 'box', 'kde', 'area', 'scatter', 'hexbin', 'pie'
#
# ax = data.plot.scatter(x='A', y='B',color='DarkBlue', label='class 1')
#
# #data.plot.scatter(x='A', y='C', color='DarkGreen', label='Class 2', ax=ax)
# plt.show()