import os
import time
import shutil
import numpy as np
import tensorflow as tf
from config import cfg

def Pyramid(input,filters=4):

    pyramid_1_model_pooling = tf.keras.layers.AveragePooling2D(16)(input)
    pyramid_1_model_conv = tf.keras.layers.Conv2D(filters=filters, kernel_size=1, strides=1)(pyramid_1_model_pooling)
    pyramid_1_model_upsampling = tf.keras.layers.UpSampling2D(16, interpolation='bilinear')(pyramid_1_model_conv)

    pyramid_2_model_pooling = tf.keras.layers.AveragePooling2D(8)(input)
    pyramid_2_model_conv = tf.keras.layers.Conv2D(filters=filters, kernel_size=1, strides=1)(pyramid_2_model_pooling)
    pyramid_2_model_upsampling = tf.keras.layers.UpSampling2D(8, interpolation='bilinear')(pyramid_2_model_conv)

    pyramid_3_model_pooling = tf.keras.layers.AveragePooling2D(4)(input)
    pyramid_3_model_conv = tf.keras.layers.Conv2D(filters=filters, kernel_size=1, strides=1)(pyramid_3_model_pooling)
    pyramid_3_model_upsampling = tf.keras.layers.UpSampling2D(4, interpolation='bilinear')(pyramid_3_model_conv)

    pyramid_4_model_pooling = tf.keras.layers.AveragePooling2D(2)(input)
    pyramid_4_model_conv = tf.keras.layers.Conv2D(filters=filters, kernel_size=1, strides=1)(pyramid_4_model_pooling)
    pyramid_4_model_upsampling = tf.keras.layers.UpSampling2D(2, interpolation='bilinear')(pyramid_4_model_conv)

    pyramid_5_model_pooling = tf.keras.layers.AveragePooling2D(32)(input)
    pyramid_5_model_conv = tf.keras.layers.Conv2D(filters=filters, kernel_size=1, strides=1)(pyramid_5_model_pooling)
    pyramid_5_model_upsampling = tf.keras.layers.UpSampling2D(32, interpolation='bilinear')(pyramid_5_model_conv)

    pyramid_6_model_pooling = tf.keras.layers.AveragePooling2D(64)(input)
    pyramid_6_model_conv = tf.keras.layers.Conv2D(filters=filters, kernel_size=1, strides=1)(pyramid_6_model_pooling)
    pyramid_6_model_upsampling = tf.keras.layers.UpSampling2D(64, interpolation='bilinear')(pyramid_6_model_conv)

    pyramid_7_model_pooling = tf.keras.layers.AveragePooling2D(128)(input)
    pyramid_7_model_conv = tf.keras.layers.Conv2D(filters=filters, kernel_size=1, strides=1)(pyramid_7_model_pooling)
    pyramid_7_model_upsampling = tf.keras.layers.UpSampling2D(128, interpolation='bilinear')(pyramid_7_model_conv)

    # Concatenate layers
    pyramid_pooling = tf.keras.layers.Concatenate()([input, pyramid_1_model_upsampling, pyramid_2_model_upsampling,
                                                     pyramid_3_model_upsampling, pyramid_4_model_upsampling,
                                                     pyramid_5_model_upsampling, pyramid_6_model_upsampling,
                                                     pyramid_7_model_upsampling])

    return pyramid_pooling

def Mynet(inputs, train=True):

    pyramid_pooling = Pyramid(inputs)
    X = tf.keras.layers.Conv2D(8, (3, 3), padding='same', activation='relu')(inputs)
    X = tf.keras.layers.Conv2D(16, (3, 3), padding='same', strides=2, activation='relu')(X)
    X = tf.keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu')(X)
    X = tf.keras.layers.Conv2D(16, (3, 3), padding='same', strides=2, activation='relu')(X)
    X = tf.keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu')(X)
    X = tf.keras.layers.Conv2D(32, (3, 3), padding='same', strides=2, activation='relu')(X)
    X = tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu')(X)

    X = tf.keras.layers.Conv2D(32, (3, 3), padding='same', strides=2, activation='relu')(X)
    X = tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu')(X)
    imgConv = tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu')(X)

    X = tf.keras.layers.UpSampling2D(2, interpolation='bilinear')(imgConv)
    X = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')(X)
    X = tf.keras.layers.UpSampling2D(2, interpolation='bilinear')(X)
    X = tf.keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu')(X)
    X = tf.keras.layers.UpSampling2D(2, interpolation='bilinear')(X)
    X = tf.keras.layers.Conv2D(9, (3, 3), padding='same', activation='relu')(X)
    X = tf.keras.layers.UpSampling2D(2, interpolation='bilinear')(X)
    imgDeconv = tf.keras.layers.Conv2D(3, (3, 3), padding='same', activation='relu')(X)

    X = tf.keras.layers.UpSampling2D(2, interpolation='bilinear')(imgConv)
    X = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')(X)
    X = tf.keras.layers.UpSampling2D(2, interpolation='bilinear')(X)
    X = tf.keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu')(X)
    X = tf.keras.layers.UpSampling2D(2, interpolation='bilinear')(X)
    X = tf.keras.layers.Conv2D(9, (3, 3), padding='same', activation='relu')(X)
    X = tf.keras.layers.UpSampling2D(2, interpolation='bilinear')(X)
    X = tf.keras.layers.Conv2D(8, (3, 3), padding='same', activation='relu')(X)

    X = tf.keras.layers.Concatenate()([X, pyramid_pooling])

    X = tf.keras.layers.Conv2D(8, (3, 3), padding='same', activation='relu')(X)
    if train:
        imgLab = tf.keras.layers.Conv2D(cfg.POINTS_NUMBER, (3, 3), padding='same', activation='sigmoid')(X)
    else:
        imgLab = tf.keras.layers.Conv2D(cfg.POINTS_NUMBER, (3, 3), padding='same', activation='sigmoid', name='out')(X)

    return [imgDeconv, imgLab]

