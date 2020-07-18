#%%
import tensorflow as tf

x = tf.constant([1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0])
f = tf.constant([1,2,3,3],dtype=tf.float32)

rx = tf.reshape(x,shape=[-1,3,3,2])


yy = tf.nn.dropout(rx,0.1)
print(yy)

# %%
