#%%
import tensorflow as tf
import numpy as np 



# %%
tensor = [[1, 2], [3, 4], [5, 6]]
mask = np.array([True, False, True])
tf.boolean_mask(tensor, mask)  # [[1, 2], [5, 6]]

# %%
src = tf.constant([[[1,2,2], [1,1,1]],[[1,1,1], [1,1,1]],[[1,2,1], [1,2,2]]],dtype=tf.float32)
mask = tf.constant([[[0,1,1], [0,0,0]],[[0,0,0], [0,0,0]],[[0,1,0], [0,1,1]]],dtype=tf.float32)
He = tf.reduce_sum(mask,axis = [1,2])
Hemask = He > 0.5
srcZ = tf.boolean_mask(src, Hemask)
maskZ = tf.boolean_mask(mask, Hemask)
HeZ = tf.boolean_mask(He, Hemask)
Zl = tf.reduce_sum(srcZ * maskZ, axis = [1,2]) / HeZ
Fl = tf.reduce_sum(srcZ * (1-maskZ), axis = [1,2]) / (2*3-HeZ)
print(Zl, Fl)

Hemask = He < 0.5
srcZ = tf.boolean_mask(src, Hemask)
Fl2 = tf.reduce_sum(srcZ, axis = [1,2]) / (2*3)
print(Fl2)

Fl = tf.concat([Fl, Fl2], 0)
l = tf.reduce_mean(Fl) + tf.reduce_mean(Zl)
print(Zl, Fl)
print(l)
#%%

src = tf.constant([[[1,2,2], [1,1,1]],[[1,1,1], [1,1,1]],[[1,2,1], [1,2,2]]])
mask = tf.constant([[[0,1,1], [0,0,0]],[[0,0,0], [0,0,0]],[[0,1,0], [0,1,1]]])
He = tf.reduce_sum(mask,axis = [1,2])
Hemask = He > 0

srcT =  tf.ragged.boolean_mask(src, (mask > 0))
srcMean =  tf.reduce_mean(srcT,axis = [1,2])
tf.reduce_mean(srcMean)

# %%
tensor = [[1, 2], [3, 4], [5, 6]]
mask = np.array([False, False, False])
tf.reduce_sum(tf.boolean_mask(tensor, mask))

# %%
src = tf.constant([[[1,2,2], [1,1,1]],[[1,1,1], [1,1,1]],[[1,2,1], [1,2,2]]],dtype=tf.float32)
tf.square(src)

# %%
