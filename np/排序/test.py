#%%
import numpy as np

a = np.array([[1,3],[3,2],[2,1]])
a[np.lexsort(a.T[:1,:]),:]


# %%
a[np.lexsort(a.T)] # 按最后一列顺序排序

# %%
# 按行
a=np.array([[1,2,1,5],[5,3,6,9],[6,2,9,5]]) 
print(a)

a1=np.lexsort(a) # 最后一行从小到大 排序的列索引
print(a1)

ind=np.lexsort(a[0:2,:])
print(ind)
A=a[:,ind]
print(A)
# %%
a=np.array([[1,2,1,5],[5,3,6,9],[6,2,9,5]])
a2=np.lexsort(a.T[:2,:])
print(a[a2,:])

# %%
import numpy as np

a = np.array([1,2,3,2])
# %%
a
# %%
np.argsort(a)
# %%
a[::-1]
# %%
