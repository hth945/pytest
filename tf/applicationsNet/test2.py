#%%
#%%
import os
import time
import shutil
import numpy as np
import tensorflow as tf
from PIL import Image
import random
import matplotlib.pyplot as plt
import cv2
from cv2 import cv2

scal = 224
sampleModel = tf.keras.applications.ResNet50V2(weights='imagenet',
                                          include_top=False,
                                          input_shape=(scal, scal, 3))
sampleModel.trianable = False
for l in sampleModel.layers:
    print(l.name)
    if l.name == 'conv4_block5_out':
        print(l)

#%%

c=[]
name=['conv2_block2_out','conv3_block3_out','conv4_block5_out','conv5_block3_out']
i=0
for l in sampleModel.layers:
    if l.name == name[i]:
        i+=1
        print(l.name)
        c.append(l.output)
        if i == 4:
            break
print(c)
model = tf.keras.models.Model(inputs=sampleModel.input, outputs=c)
tf.keras.utils.plot_model(model, to_file='rennetRpn.png', show_shapes=True, show_layer_names=True)
#%%

model.outputs
#%%

sampleModel.layers['conv4_block5_out']
#%%
img = cv2.imread('hua.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = img/255.0
img = cv2.resize(img,(224,224))
plt.imshow(img)
o = sampleModel(np.expand_dims(img,0))
# %%
probs = tf.nn.softmax(o)
probs=probs.numpy()
np.max(probs)
# %%
np.argmax(probs)
# %%
probs
# %%
print('Predicted:', tf.keras.applications.resnet_v2.decode_predictions(o, top=3)[0])
# %%
img.shape
# %%
w = sampleModel.get_weights()
w[0]
# %%

for l in sampleModel.layers:
    print(l.name)
# %%

# %%
