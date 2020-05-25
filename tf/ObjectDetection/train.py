import os
import time
import shutil
import numpy as np
import tensorflow as tf
from config import cfg
import mynet
from VRDatasetC import NpDataset



size = cfg.TRAIN.INPUT_SIZE
inputs = tf.keras.Input(shape=(size, size, 3))
imgDeconv, imgLab = mynet.Mynet(inputs)
model = tf.keras.Model(inputs, [imgDeconv, imgLab])


logdir = "./data/log"
optimizer = tf.keras.optimizers.Adam(0.0002)
if os.path.exists(logdir):
    shutil.rmtree(logdir)
writer = tf.summary.create_file_writer(logdir)

# global_steps = tf.Variable(1, trainable=False, dtype=tf.int64)


@tf.function
def learningModel(image_data, target):
    with tf.GradientTape() as tape:
        target = tf.cast(target, tf.float32)
        image_data = image_data / 255.0
        img, lab = model(image_data, training=True)

        meanLoss = tf.reduce_mean(tf.keras.losses.mean_squared_error(img, image_data)) / 50# / 5000.0

        target2 = target*0.9 + 0.05
        softmaxLoss = tf.reduce_mean(tf.keras.losses.mean_squared_error(tf.expand_dims(target2, -1), lab) * (target * 10 + 1))

        # y_onehot = tf.one_hot(target, depth=2)
        # softmaxLoss = tf.keras.losses.categorical_crossentropy(y_onehot, lab, from_logits=True)


        #
        # softmaxLoss = softmaxLoss * (target * 2 + 1)

        softmaxLoss = tf.reduce_mean(softmaxLoss)
        # print(softmaxLoss,meanLoss)
        loss_regularization = []
        for p in model.trainable_variables:
            loss_regularization.append(tf.nn.l2_loss(p))
        loss_regularization = tf.reduce_sum(tf.stack(loss_regularization)) * 0.00002

        total_loss = softmaxLoss + meanLoss  #+ loss_regularization
        gradients = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        # tf.print("lr: %.6f  softmaxLoss: %4.5f  meanLoss: %4.5f total_loss: %4.5f" %
        #          (optimizer.lr.numpy(), softmaxLoss, meanLoss, total_loss))
        tf.print(softmaxLoss, meanLoss, loss_regularization)
        # global_steps
        # # writing summary data
        # with writer.as_default():
        #     tf.summary.scalar("lr", optimizer.lr, step=global_steps)
        #     tf.summary.scalar("loss/total_loss", total_loss, step=global_steps)
        #     tf.summary.scalar("loss/meanLoss", meanLoss, step=global_steps)
        #     tf.summary.scalar("loss/softmaxLoss", softmaxLoss, step=global_steps)
        # writer.flush()


global_steps = 0
model.load_weights('oldModel.h5')
trainset = NpDataset('train')
for epoch in range(cfg.TRAIN.EPOCHS):
    for image_data, target in trainset:
        learningModel(image_data, target)
        global_steps += 1
        tf.print(global_steps)
    model.save("oldModel.h5")
    tf.print("saveModel")


