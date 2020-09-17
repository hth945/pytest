#%%
import numpy as np
def softmax(x):
    x = np.exp(x)/np.sum(np.exp(x))
    return x
print(softmax([2,3]))


# %%
