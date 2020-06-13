#%%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.Series([1,3,5,np.nan,6,8])

# %%
dates = pd.date_range('20130101', periods=6)
df = pd.DataFrame(np.random.randn(6,4), index=dates, columns=list('ABCD'))
df[df > 0]
# %%
df = pd.DataFrame({ 'A' : list(range(4)), 
                        'B': pd.Timestamp('20130102'),
                        'C': pd.Series(1,index=list(range(4)), dtype='float32'),
                        'D': np.array([3]*4),
                        'E': pd.Categorical(['test','test2','test','test2']),
                        'F': "foo"})


# %%
print(df.head())
print(df.tail(3))
print(df.index)
print(df.columns)
print(df.dtypes)
print(df.values)


# %%
print(df.describe()) # max mean std 等分析


# %%
df.sort_index(axis=1, ascending=False)

# %%
df.sort_values(by='E')

# %%
df[0:2]

# %%
df.loc[0:1,['A','B']] # 按键索引
df.loc[2,'A']
df.at[2,'A']

# %%
df.iloc[0:1,0:2] # 按编号索引
df.iat[1,1]

# %%
df[df.A > 1]  # 按bool索引

# %%
df2 = df.copy()
df2['E'] = ['one', 'one','two','three','four','three']
df2[df2['E'].isin(['two','four'])]
# %%
s1 = pd.Series([1,2,3,4,5,6], index=pd.date_range('20130102', periods=6))
df['F'] = s1
df
# %%
s1 = pd.Series([1,2,3,4,5,6],index=df.index)
df['F'] = s1
df

# %%
df.at[dates[0],'A'] = 0
df.loc[:,'D'] = np.array([5] * len(df))
df
# %%
df2 = df.copy()
df2[df2 > 0] = -df2
df2
# %%
df1 = df.reindex(index=dates[0:4], columns=list(df.columns) + ['E'])
df1.loc[dates[0]:dates[1],'E'] = 1
print(df1)
print(df1.dropna(how='any'))
print( df1.fillna(value=5))
print(pd.isnull(df1))
# %%
s = pd.Series(np.random.randint(0, 7, size=10))
s.value_counts()

#%%
s = pd.Series([1,3,5,np.nan,6,8], index=dates).shift(2)
df.sub(s, axis='index')

# %%

df
# %%
df.apply(print)

# %%
df['F'].apply(print)

# %%
s = pd.Series([1,3,5,np.nan,6,8], index=dates).shift(2)
def convertRate(row):
    print(type(row))
    return row
s.apply(convertRate)

# %%
print(type(2))
def aaa(a):
    print(type(a))
aaa(1)
# %%
type(1)

# %%
