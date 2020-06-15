#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

test_dict = {'id':[1,2,3,4,5,6],
             'data':[1,1,2,3,3,3],
             'data2':[1,1,1,1,1,1]}
test_dict_df = pd.DataFrame(test_dict)


# %%
test_dict_df.groupby(by="data").size()
#%%
test_dict_df.groupby(by="data").size().plot()
plt.show()
test_dict_df.groupby(by="data").size().plot.bar()
plt.show()
test_dict_df.groupby(by="data").size().plot.barh()
plt.show()
test_dict_df.groupby(by="data").size().hist()
plt.show()
# %%
grouped = test_dict_df.groupby(by="data")
print(grouped.get_group(3))

# %%
df = grouped.get_group(3).reset_index()
print(df)

# %%
grouped.count()

# %%
test_dict_df['data'].hist()
plt.show()
test_dict_df[['data','data2']].plot.hist(alpha=0.5,bins=20)  # ; 交叉直方图
plt.show()

# %%
# 分组聚合
# def getSum(data):
#     total = 0
#     for d in data:
#         total+=d
#     return total
# print(grouped.aggregate(np.median))
# print(grouped.aggregate({'Age':np.median, 'Score':np.sum}))
# print(grouped.aggregate({'Age':getSum}))
grouped.aggregate({'id':np.sum, 'data2':np.sum})

# %%
