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
