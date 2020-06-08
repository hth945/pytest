import os, glob

import numpy as np
import tensorflow as tf
from myDataset import *
from myconfig import cfg
from yoloNet import *


model = yoloNetModle()
tf.keras.utils.plot_model(model, show_shapes=True, show_layer_names=True)
loadWeight(model,'yolo.weights')


train_db = get_dataset(r'..\..\dataAndModel\data\杂草识别\data\train\image', r'..\..\dataAndModel\data\杂草识别\data\train\annotation', 10, cfg.TRAIN.OBJ_NAMES)
aug_train_db = augmentation_generator(train_db)
train_gen = ground_truth_generator(aug_train_db)


def train(epoches):
    optimizer = keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.9,
                                      beta_2=0.999, epsilon=1e-08)

    for epoch in range(epoches):

        for step in range(30):
            img, detector_mask, matching_true_boxes, matching_classes_oh, true_boxes = next(train_gen)
            with tf.GradientTape() as tape:
                y_pred = model(img, training=True)
                loss, sub_loss = yolo_loss(detector_mask, \
                                           matching_true_boxes, matching_classes_oh, \
                                           true_boxes, y_pred)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            print(epoch, step, float(loss), float(sub_loss[0]), float(sub_loss[1]), float(sub_loss[2]))


# %%
train(10)
model.save_weights('weights/epoch10.ckpt')



