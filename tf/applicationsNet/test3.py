#%%
import tensorflow as tf
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
model = tf.keras.applications.ResNet50V2(weights='imagenet',
                                          include_top=True,
                                          input_shape=(scal, scal, 3))

# img_path = 'gou.jpg'
img_path = 'mao.bmp'
img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
x = tf.keras.preprocessing.image.img_to_array(img)
x = np.expand_dims(x, axis=0)
# x = x[...,[2,1,0]]
preds = model.predict(x)
# 将结果解码为元组列表 (class, description, probability)
# (一个列表代表批次中的一个样本）
print('Predicted:', tf.keras.applications.resnet_v2.decode_predictions(preds, top=4)[0])
# Predicted: [(u'n02504013', u'Indian_elephant', 0.82658225), (u'n01871265', u'tusker', 0.1122357), (u'n02504458', u'African_elephant', 0.061040461)]
# %%
preds
# %%
np.max(preds)
# %%
np.argmax(preds)
# %%
preds
# %%
x.shape
# %%
