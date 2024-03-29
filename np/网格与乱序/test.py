#%%
import numpy as np

x,y = np.mgrid[0:360:1,0.5:1:0.1]
grid = np.c_[x.ravel(), y.ravel()]
print(grid)


# %%
x = np.array([0, 1, 2])
y = np.array([0, 1])

X, Y = np.meshgrid(x, y)
print(X)
print(Y)

# %%
np.mgrid[1:3:1]

# %%
np.random.shuffle(grid)
grid
# %%
