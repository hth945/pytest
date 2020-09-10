#%%

import cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt 
from cv2 import cv2

img = np.ones([1,10,10,1])
img[0,5,0,:] = 0

b = tf.image.crop_and_resize(img,[[0,0,1,1],[0.2,0.6,1.3,0.9]],box_indices=[0,0],crop_size=(3,3))




# %%
b
# %%
