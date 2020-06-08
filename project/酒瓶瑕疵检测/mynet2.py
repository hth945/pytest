import os
import time
import shutil
import numpy as np
import tensorflow as tf
from config import cfg
import tensorflow_hub as hub


# resnet50 = tf.saved_model.load('bit_m-r50x1_1') # in [244,244,3] out
#
#
# # for v in resnet50.trainable_variables:
# #     print(v.name, '  ', v.shape)
# # print('')
# # for v in resnet50.trainable_variable_ids:
# #     print(v.name, '  ', v.shape)
#
# # for v in imported.variables:
# #     print(v.name, '  ', v.shape)
# print(len(resnet50.trainable_variables))
# print(len(resnet50.variables))
# # %%
#
# a = tf.zeros([1,224,224,3])
# # a = tf.zeros([224,224,3])
# o = resnet50(a)
# print(o.shape)



# resnet50 = hub.KerasLayer('bit_m-r50x1_1')
# model = tf.keras.Sequential([
#     resnet50,
# ])
# model.build((None,224,224,3))
# model.summary()

resnet50 = hub.KerasLayer('bit_m-r50x1_1')
model = tf.keras.Sequential([
    resnet50,
])
model.build((None,)+(224,224)+(3,))
model.summary()
#%%
o = model(tf.zeros([1,224,224,3]))
print(o.shape)
# tf.keras.utils.plot_model(model, show_shapes=True, show_layer_names=True)