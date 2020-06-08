#%%
import os, glob

import numpy as np
import tensorflow as tf
from myDataset import *
from myconfig import cfg
from yoloNet import *

model = yoloNetModle()
tf.keras.utils.plot_model(model, show_shapes=True, show_layer_names=True)

weight_reader = WeightReader('yolo.weights')

print(weight_reader.all_weights[:4])

conv_layer = model.get_layer('conv_1')
for i in conv_layer.weights:
    print(i.shape)

norm_layer = model.get_layer('norm_1')
for i in norm_layer.weights:
    print(i.shape)
    print(i.name)


# %%
for i in norm_layer.weights:
    print(i.shape)
    print(i.name)

# %%

bn1 = layers.BatchNormalization(name='bn1')

inTem = tf.convert_to_tensor([1,2,3.0])

outTem = bn1(inTem)
print(outTem)

bn1.weights[1].assign(np.array([1,4,9], dtype=np.float32))

outTem = bn1(inTem)
print(outTem)

for i in bn1.weights:
    print(i)
    # print(i.shape)
    # print(i.name)

# %%

# %%
