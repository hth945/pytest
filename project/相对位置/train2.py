import os
import time
import shutil
import numpy as np
import tensorflow as tf
from config import cfg
import mynet
from VRDatasetC import NpDataset
import tensorflow_hub as hub


scal = cfg.TRAIN.INPUT_SIZE
xception = tf.keras.applications.Xception(weights='imagenet',
                                          include_top=False,
                                          input_shape=(scal, scal, 3))
xception.trianable = False

inputs = tf.keras.layers.Input(shape=(scal, scal, 3))
x = xception(inputs)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(2048, activation='relu')(x)
x = tf.keras.layers.Dense(256, activation='relu')(x)
out = tf.keras.layers.Dense(3)(x)
model = tf.keras.models.Model(inputs=inputs, outputs=out)
# , activation='sigmoid'

##############my######################
# size = cfg.TRAIN.INPUT_SIZE
# inputs = tf.keras.Input(shape=(size, size,3))
# imgLab = mynet.Mynet(inputs)
# model = tf.keras.Model(inputs,  imgLab)

###############hub#################################
# resnet50 = hub.KerasLayer('../../dataAndModel/model/bit_m-r50x1_1')
# model = tf.keras.Sequential()
# model.add(resnet50)
# model.add(tf.keras.layers.Dense(512, activation='relu'))
# model.add(tf.keras.layers.Dense(3, activation='sigmoid'))
# model.build(input_shape=(None,224,224,3))
# model.summary()

print('initOK')
tf.keras.utils.plot_model(model, show_shapes=True, show_layer_names=True)

logdir = "./data/log"
optimizer = tf.keras.optimizers.Adam(0.0002)
if os.path.exists(logdir):
    shutil.rmtree(logdir)

writer = tf.summary.create_file_writer(logdir)

# global_steps = tf.Variable(1, trainable=False, dtype=tf.int64)

import cv2
# @tf.function
def learningModel(image_data, target):
    with tf.GradientTape() as tape:
        lab = model(image_data, training=True)
        t = target[:,9:12] # 0,1

        meanLoss = tf.reduce_mean(t * tf.square(tf.nn.relu(0.8-lab)) + (1-t)*tf.square(tf.nn.relu(lab - 0.2)))

        total_loss = meanLoss
        gradients = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        tf.print(meanLoss)

global_steps = 0
# model.load_weights('oldModel.h5')
trainset = NpDataset('train')
for epoch in range(cfg.TRAIN.EPOCHS):
    for image_data, target in trainset:
        learningModel(image_data, target)
        global_steps += 1
        tf.print(global_steps)
    model.save("oldModel3.h5")
    tf.print("saveModel")


