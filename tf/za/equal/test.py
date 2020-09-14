#%%
import tensorflow as tf
import numpy as np


t = tf.constant([0,1,2,1])

# %%
tf.equal(t, 1)
# %%
tf.cast(tf.equal(t, 1), tf.int32)
# %%t
indices = tf.where(tf.not_equal(t, 1))