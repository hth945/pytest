import os
import time
import shutil
import numpy as np
import tensorflow as tf
from config import cfg
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

def Pyramid(input,filters=4):

    pyramid_1_model_pooling = tf.keras.layers.AveragePooling2D(16)(input)
    pyramid_1_model_conv = tf.keras.layers.Conv2D(filters=filters, kernel_size=1, strides=1)(pyramid_1_model_pooling)
    #  pyramid_1_model_upsampling = tf.compat.v1.image.resize_bilinear(pyramid_1_model_conv, [input.shape[1], input.shape[2]])
    pyramid_1_model_upsampling = tf.keras.layers.UpSampling2D(16, interpolation='bilinear')(pyramid_1_model_conv)

    pyramid_2_model_pooling = tf.keras.layers.AveragePooling2D(8)(input)
    pyramid_2_model_conv = tf.keras.layers.Conv2D(filters=filters, kernel_size=1, strides=1)(pyramid_2_model_pooling)
    # pyramid_2_model_upsampling = tf.compat.v1.image.resize_bilinear(pyramid_2_model_conv, [input.shape[1], input.shape[2]])
    pyramid_2_model_upsampling = tf.keras.layers.UpSampling2D(8, interpolation='bilinear')(pyramid_2_model_conv)

    pyramid_3_model_pooling = tf.keras.layers.AveragePooling2D(4)(input)
    pyramid_3_model_conv = tf.keras.layers.Conv2D(filters=filters, kernel_size=1, strides=1)(pyramid_3_model_pooling)
    # pyramid_3_model_upsampling = tf.compat.v1.image.resize_bilinear(pyramid_3_model_conv, [input.shape[1], input.shape[2]])
    pyramid_3_model_upsampling = tf.keras.layers.UpSampling2D(4, interpolation='bilinear')(pyramid_3_model_conv)

    pyramid_4_model_pooling = tf.keras.layers.AveragePooling2D(2)(input)
    pyramid_4_model_conv = tf.keras.layers.Conv2D(filters=filters, kernel_size=1, strides=1)(pyramid_4_model_pooling)
    # pyramid_4_model_upsampling = tf.compat.v1.image.resize_bilinear(pyramid_4_model_conv,[input.shape[1], input.shape[2]])
    pyramid_4_model_upsampling = tf.keras.layers.UpSampling2D(2, interpolation='bilinear')(pyramid_4_model_conv)

    # Concatenate layers
    pyramid_pooling = tf.keras.layers.Concatenate()([input, pyramid_1_model_upsampling, pyramid_2_model_upsampling,
                                                     pyramid_3_model_upsampling, pyramid_4_model_upsampling])

    return pyramid_pooling

def Mynet(inputs, train=True):

    # pyramid_pooling = Pyramid(inputs)
    X = tf.keras.layers.Conv2D(16, (3, 3), padding='same', strides=2, activation='relu')(inputs) # 112
    X = tf.keras.layers.Conv2D(16, (3, 3), padding='same', strides=2, activation='relu')(X)      # 56
    X = tf.keras.layers.Conv2D(32, (3, 3), padding='same', strides=2, activation='relu')(X)      # 28
    X = tf.keras.layers.Conv2D(64, (3, 3), padding='same', strides=2, activation='relu')(X)      # 14
    X = tf.keras.layers.Conv2D(64, (3, 3), padding='same', strides=2, activation='relu')(X)      # 7
    X = tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')(X)
    X = tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')(X)


    X = tf.keras.layers.Conv2D(256, (7, 7), padding='valid', activation='relu')(X)
    if train:
        imgLab = tf.keras.layers.Conv2D(3, (1, 1), padding='same', activation='sigmoid')(X)
    else:
        imgLab = tf.keras.layers.Conv2D(3, (1, 1), padding='same', activation='sigmoid', name='out')(X)

    return imgLab

