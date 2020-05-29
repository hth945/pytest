#%%
import numpy as np
import random
import tensorflow as tf

t1 = tf.constant([[1, 2, 3], [4, 5, 6]])
t2 = tf.constant([[7, 8, 9], [10, 11, 12]])
tf.concat([t1, t2], 0)  # [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
tf.concat([t1, t2], 1)  # [[1, 2, 3, 7, 8, 9], [4, 5, 6, 10, 11, 12]]

# %%
def layers(inputs,units,activation=None):
    l = tf.keras.layers.Dense(units,activation=activation)(inputs)
    print(inputs.shape)
    print(l.shape)
    x = tf.concat([inputs, l], 1)
    return l, x

tf.keras.backend.clear_session()

inputs = tf.keras.Input(shape=(25))
_,x = layers(inputs,50, "relu")
_,x = layers(x,50, "relu")
l1,x = layers(x,25)
_,x = layers(x,50, "relu")
l2,x = layers(x,25, "relu")

model = tf.keras.Model(inputs, [l1, l2])
tf.keras.utils.plot_model(model, show_shapes=True, show_layer_names=True)

# %%