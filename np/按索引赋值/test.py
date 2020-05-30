#%%
import numpy as np

i = np.array([0,2])
j = np.array([0,1])
one = np.ones([3,3])
# %%
tem = np.zeros([3,3])
i,j = np.mgrid[0:3:2,0:3:2]
tem[i,j] = one[i,j]
print(tem)
#%%

tem = np.zeros([3,3])
x = np.array([0, 1, 2])
y = np.array([0, 1])
X, Y = np.meshgrid(x, y)
X = X.ravel()
Y = Y.ravel()
tem[X,Y] = one[X,Y]
print(tem)




# %%
