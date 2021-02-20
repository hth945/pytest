#%%
import matplotlib.pyplot as plt
import numpy as np

x=np.array([[0.295,0.330],[0.315,0.330],[0.295,0.290],[0.275,0.290],[0.295,0.330]])
plt.plot(0.3077, 0.3147,".", color = "r")
plt.plot(x[:,0],x[:,1])
plt.show()
# %%
