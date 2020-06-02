#%%
import tensorflow as tf

c = tf.constant([[3.,6], [5.,6], [6.,7]])

temMax = tf.expand_dims(tf.reduce_max(c, 1),1)
print(temMax)
temMin = tf.expand_dims(tf.reduce_min(c, 1),1)
print(temMin)
o = (c - temMin)/ (temMax - temMin)

print(o)
#%%

test = tf.constant([[3.,4], [5.,6], [6.,7]])
# s = tf.constant([[1.],[2],[3]])
s = tf.constant([1,2,3.])
n = test - tf.expand_dims(s,1)
print(n)


# %%
