import os
import time
import shutil
import numpy as np
import tensorflow as tf
from config import cfg
import mynet
from VRDatasetC import NpDataset
import cv2


size = cfg.TRAIN.INPUT_SIZE
inputs = tf.keras.Input(shape=(size, size, 3))
imgDeconv, imgLab = mynet.Mynet(inputs)
model = tf.keras.Model(inputs, [imgDeconv, imgLab])
model.load_weights('oldModel.h5')


train_dateset = NpDataset('train')
# for imgs, lables in train_dateset:
#     # print(lables[3])
#     # print(imgs.shape)
#     # print(lables.shape)
#     # print(np.unique(imgs[0]))
#
#     y_onehot = tf.one_hot(lables, depth=3)
#     print(y_onehot.shape)
#     cv2.imshow('y_onehot', y_onehot[3].numpy() * 1.0)
#     cv2.imshow('1', imgs[3] / 255.0)
#     cv2.imshow('2', lables[3] * 0.5)
#
#     img, lab = model(imgs)
#     outPuts = tf.nn.softmax(lab)
#     cv2.imshow('outPuts', outPuts[3].numpy() * 1.0)
#
#     cv2.waitKey(0)


for imgs, lables in train_dateset:
    y_onehot = tf.one_hot(lables, depth=3)
    print(y_onehot.shape)
    cv2.imshow('y_onehot', y_onehot[3].numpy() * 1.0)
    cv2.imshow('1', imgs[3] / 255.0)
    cv2.imshow('2', lables[3] * 0.5)

    imgs = imgs / 255.0
    img, lab = model(imgs)
    # outPuts = tf.nn.softmax(lab)
    cv2.imshow('outPuts', lab[3][:,:,0].numpy() * 1.0)

    cv2.waitKey(0)

