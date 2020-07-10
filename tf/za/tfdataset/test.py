#%%
import tensorflow as tf
import time
import cv2

def myfun(x):
    img = cv2.imread('123.png')
    return x+1

dataset = tf.data.Dataset.range(1, 100)  # ==> [ 1, 2, 3, 4, 5 ]
#dataset = dataset.map(myfun, num_parallel_calls=3) # , deterministic=False
dataset = dataset.map(myfun) 
for i in dataset:
    print(i)

# %%


import itertools

def gen():
    for i in itertools.count(1):
        time.sleep(1)
        yield (i, [1] * i)

dataset = tf.data.Dataset.from_generator(
     gen,
     (tf.int64, tf.int64),
     (tf.TensorShape([]), tf.TensorShape([None])))
n = 0
for i in dataset:
    print(i)
    n += 1
    if n > 10:
        break


# %%
