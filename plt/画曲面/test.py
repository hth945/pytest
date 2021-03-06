#%%
import matplotlib.pyplot as plt
import numpy as np
x = np.arange(0,25)
y = x*2
plt.plot(x,y, label='line 1')
plt.legend()
plt.show()

# %%
# 曲面图
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X = [1, 1, 2, 2]
Y = [3, 4, 4, 3]
Z = [1, 2, 1, 1]
ax.plot_trisurf(X, Y, Z)
plt.show()

# %%
#散点图
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X = [1, 1, 2, 2]
Y = [3, 4, 4, 3]
Z = [1, 2, 1, 1]
ax.scatter(X, Y, Z)
plt.show()

# %%
