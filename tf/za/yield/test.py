#%%

import numpy as np

 
def data_generator():
    dataset = np.array(range(2))
    for d in dataset:
        yield d
# %%
for i in data_generator():
    print(i)
#%%
import tensorflow as tf
dataset = tf.data.Dataset.from_generator(data_generator, (tf.int32))
dataset = dataset.repeat(3)
dataset = dataset.batch(10)
for i in dataset:
    print(i)
    break

# %%


class DataGenerator:

    def __init__(self):
        pass
    def __call__(self):
        dataset = np.array(range(2))
        for d in dataset:
            yield d
t = DataGenerator()
for i in t():
    print(i)

# %%
