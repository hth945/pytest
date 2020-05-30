#%%
import tensorflow as tf

# %%
b = tf.constant([[1, 2], [3, 4]])
tf.tile(b, (2, 1))
# %%
t1 = [[1, 2, 3], [4, 5, 6]]
t2 = [[7, 8, 9], [10, 11, 12]]
tf.concat([t1, t2],0)

# %%

lin = tf.expand_dims(tf.linspace(0.,1.,25), 0)
tf.tile(lin, (2, 1))

# %%
