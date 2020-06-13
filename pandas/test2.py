#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1234)
d1 = pd.Series(2*np.random.normal(size = 100)+3)
d2 = np.random.f(2,4,size = 100)
d3 = np.random.randint(1,100,size = 100)

print('非空元素计算: ', d1.count()) #非空元素计算
print('最小值: ', d1.min()) #最小值
print('最大值: ', d1.max()) #最大值
print('最小值的位置: ', d1.idxmin()) #最小值的位置，类似于R中的which.min函数
print('最大值的位置: ', d1.idxmax()) #最大值的位置，类似于R中的which.max函数
print('10%分位数: ', d1.quantile(0.1)) #10%分位数
print('求和: ', d1.sum()) #求和
print('均值: ', d1.mean()) #均值
print('中位数: ', d1.median()) #中位数
print('众数: ', d1.mode()) #众数
print('方差: ', d1.var()) #方差
print('标准差: ', d1.std()) #标准差
print('平均绝对偏差: ', d1.mad()) #平均绝对偏差
print('偏度: ', d1.skew()) #偏度
print('峰度: ', d1.kurt()) #峰度
print('描述性统计指标: ', d1.describe()) #一次性输出多个描述性统计指标

# %%
def stats(x):
	return pd.Series([x.count(),x.min(),x.idxmin(),x.quantile(.25),x.median(),x.quantile(.75),
                      x.mean(),x.max(),x.idxmax(),x.mad(),x.var(),x.std(),x.skew(),x.kurt()],
                     index = ['Count','Min','Whicn_Min','Q1','Median','Q3','Mean','Max',
                              'Which_Max','Mad','Var','Std','Skew','Kurt'])
print(stats(d1))

# %%
# Series的层次化索引，索引是一个二维数组，相当于两个索引决定一个值
# 有点类似于DataFrame的行索引和列索引
s = pd.Series(np.arange(1,10),index=[["a","a","a","b","b","c","c","d","d"],[1,2,3,1,2,3,1,2,3]])
print(s)
print(s.index)
#选取外层索引为a的数据
print(s['a'])
#选取外层索引为a和内层索引为1的数据
print(s['a',1])
#选取外层索引为a和内层索引为1,3的数据
print(s['a'][[1,3]])
#层次化索引的切片，包括右端的索引
print(s[['a','c']])
print(s['b':'d'])
#通过unstack方法可以将Series变成一个DataFrame
#数据的类型以及数据的输出结构都变成了DataFrame，对于不存在的位置使用NaN填充
print(s.unstack())
# %%
data = pd.DataFrame(np.random.randint(0,150,size=(8,12)),
               columns = pd.MultiIndex.from_product([['模拟考','正式考'],
                                                   ['数学','语文','英语','物理','化学','生物']]),
               index = pd.MultiIndex.from_product([['期中','期末'],
                                                   ['雷军','李斌'],
                                                  ['测试一','测试二']]))
data

# %%
data['模拟考'][['语文','数学']]

# %%
complaints['Complaint Type'].value_counts()