#%%
import json
import cv2
from cv2 import cv2
import numpy as np
import os
from config import cfg

imgs = np.zeros([1, cfg.TRAIN.INPUT_SIZE[0], cfg.TRAIN.INPUT_SIZE[1],3], dtype=np.uint8)

for name in os.listdir('datalab2/images'):
    img = cv2.imread('datalab2/images/' + name)
    if img.shape[0] != 492:
        continue
    imgs[0,:492, :, :] = img[:, :656]
    print(img.shape[0])

    import tensorflow as tf
    modelTest = tf.keras.models.load_model('oldModel.h5')

    img, lab = modelTest(imgs / 255.0)

    cv2.imshow('imgsSrc', imgs[0] / 255.0)
    cv2.imshow('lab', lab[0].numpy())

    cv2.imshow("1", img[0].numpy())
    cv2.waitKey(0)



