#%%
import  os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import gym
import logging
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

tf.random.set_seed(22)
np.random.seed(22)

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
optimizer = tf.keras.optimizers.Adam(0.01)
#%%

kernel = tf.Variable([[1.1,1.0]])
i = tf.constant([[1.0,1.0]])
logits = i*kernel
probs = tf.nn.softmax(kernel)
print(probs)
#%%
for _ in range(100):
    with tf.GradientTape() as tape:
        logits = i*kernel
        probs = tf.nn.softmax(kernel)
        entropy_loss = tf.keras.losses.categorical_crossentropy(probs, probs)
        
    grads = tape.gradient(entropy_loss, [kernel])
    optimizer.apply_gradients(zip(grads, [kernel]))
print(probs)
#%%
kernel
# %%

# %%
i = tf.constant([[1.0,1.0]])
probs = tf.nn.softmax(i)
probs = tf.constant([[1.0,0.0],[1.0,0.0]])
probs2 = tf.constant([[0.9,0.1],[0.9,0.1]])
entropy_loss = tf.keras.losses.categorical_crossentropy(probs, probs2)
print(entropy_loss)
# %%
import math
0.3*math.log(0.3) + 0.7*math.log(0.7) 
# %%
import numpy as np
def softmax(x):
    x = np.exp(x)/np.sum(np.exp(x))
    return x


def ce(y_true, y_pred):
    return -(y_true[0]*math.log(y_pred[0]) + y_true[1]*math.log(y_pred[1]))
prob = softmax([2,3])
print(prob)
ls = ce(prob,prob)
print(ls)


i = tf.constant([[2.0,3.0]])
probs = tf.nn.softmax(i)
print(probs)
entropy_loss = tf.keras.losses.categorical_crossentropy(probs, probs)
print(entropy_loss)

entropy_loss = tf.keras.losses.categorical_crossentropy(i, i,from_logits=True)
print(entropy_loss)


# %%
yTrue = tf.constant([[1.0,0.0]])
# yTrue = tf.constant([[0.9,0.1]])
i = tf.constant([[30.0,30.0]])
probs = tf.nn.softmax(i)
print(probs)
entropy_loss = tf.keras.losses.categorical_crossentropy(yTrue, probs)
print(entropy_loss)

entropy_loss = tf.keras.losses.categorical_crossentropy(yTrue, i,from_logits=True)
print(entropy_loss)
# %%
