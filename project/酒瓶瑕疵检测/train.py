import os
import time
import shutil
import numpy as np
import tensorflow as tf
from tensorflow import keras
from config import cfg
import mynet
from VRDatasetC import NpDataset

size = cfg.TRAIN.INPUT_SIZE
inputs = tf.keras.Input(shape=(size[0], size[1],3))
imgDeconv, imgLab = mynet.Mynet(inputs)
model = tf.keras.Model(inputs, [imgDeconv, imgLab])
keras.utils.plot_model(model, show_shapes=True, show_layer_names=True)


logdir = "./data/log"
optimizer = tf.keras.optimizers.Adam(0.0002)
if os.path.exists(logdir):
    shutil.rmtree(logdir)
writer = tf.summary.create_file_writer(logdir)

# global_steps = tf.Variable(1, trainable=False, dtype=tf.int64)


# @tf.function
def learningModel(image_data, target):
    with tf.GradientTape() as tape:
        image_data = tf.cast(image_data, tf.float32)
        target = tf.cast(target, tf.float32)

        image_data = image_data / 255.0
        target = target / 255.0
        img, lab = model(image_data, training=True)

        meanLoss = tf.reduce_mean(tf.keras.losses.mean_squared_error(img, image_data)) / 5000.0  # / 5000.0

        target2 = target * 0.6 + 0.2
        softmaxLoss = tf.reduce_mean(
            tf.keras.losses.mean_squared_error(tf.expand_dims(target2, -1), lab) * (target * 10 + 1))

        loss_regularization = []
        for p in model.trainable_variables:
            loss_regularization.append(tf.nn.l2_loss(p))
        loss_regularization = tf.reduce_sum(tf.stack(loss_regularization)) * 0.00002

        total_loss = softmaxLoss + meanLoss  #+ loss_regularization
        gradients = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        tf.print(softmaxLoss, meanLoss, loss_regularization)


global_steps = 0
# model.load_weights('oldModel.h5')
trainset = NpDataset('train')
for epoch in range(cfg.TRAIN.EPOCHS):
    for image_data, target in trainset:
        learningModel(image_data, target)
        global_steps += 1
        tf.print(global_steps)
    model.save("oldModel.h5")
    tf.print("saveModel")


