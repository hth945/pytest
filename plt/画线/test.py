#%%
# coding: utf-8
import matplotlib.pyplot as plt
import numpy as np

x = np.arange(2,10)
y = x ** 2

plt.plot(x,y)
plt.show()

# %%
x=np.array([[0.295,0.330],[0.315,0.330],[0.295,0.290],[0.275,0.290],[0.295,0.330]])
plt.plot(0.3077, 0.3147,".", color = "r")
plt.plot(x[:,0],x[:,1])
plt.show()
# %%
