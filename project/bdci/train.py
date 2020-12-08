#%%
import os

# GPU memory garbage collection optimization flags
os.environ['FLAGS_eager_delete_tensor_gb'] = "0.0"

import sys
import argparse
import pprint
import random
import shutil
import numpy as np
import codecs
import copy

from PIL import Image
import cv2
from config import cfg
from model_builder import ModelPhase
import data_aug as aug
import myHRnet
import myDataset
import metrics

def parse_args():
    parser = argparse.ArgumentParser(description='PaddleSeg training')
    parser.add_argument(
        '--cfg',
        dest='cfg_file',
        help='Config file for training (and optionally testing)',
        default='testunet.yaml',
        type=str)
    parser.add_argument(
        '--use_gpu',
        dest='use_gpu',
        help='Use gpu or cpu',
        action='store_true',
        default=False)
    parser.add_argument(
        '--use_mpio',
        dest='use_mpio',
        help='Use multiprocess I/O or not',
        action='store_true',
        default=False)
    parser.add_argument(
        '--log_steps',
        dest='log_steps',
        help='Display logging information at every log_steps',
        default=10,
        type=int)
    parser.add_argument(
        '--debug',
        dest='debug',
        help='debug mode, display detail information of training',
        action='store_true')
    parser.add_argument(
        '--use_vdl',
        dest='use_vdl',
        help='whether to record the data during training to VisualDL',
        action='store_true')
    parser.add_argument(
        '--vdl_log_dir',
        dest='vdl_log_dir',
        help='VisualDL logging directory',
        default=None,
        type=str)
    parser.add_argument(
        '--do_eval',
        dest='do_eval',
        help='Evaluation models result on every new checkpoint',
        action='store_true')
    parser.add_argument(
        'opts',
        help='See utils/config.py for all options',
        default=None,
        nargs=argparse.REMAINDER)
    parser.add_argument(
        '--enable_ce',
        dest='enable_ce',
        help='If set True, enable continuous evaluation job.'
             'This flag is only used for internal test.',
        action='store_true')

    # NOTE: This for benchmark
    parser.add_argument(
        '--is_profiler',
        help='the profiler switch.(used for benchmark)',
        default=0,
        type=int)
    parser.add_argument(
        '--profiler_path',
        help='the profiler output file path.(used for benchmark)',
        default='./seg.profiler',
        type=str)
    return parser.parse_args()


if __name__ == '__main__':
    cfg.update_from_file("testunet.yaml")

    cfg.TRAINER_ID = int(os.getenv("PADDLE_TRAINER_ID", 0))
    cfg.NUM_TRAINERS = int(os.environ.get('PADDLE_TRAINERS_NUM', 1))

    cfg.check_and_infer()
    # print(pprint.pformat(cfg))

    dataset = myDataset.SegDataset(
        file_list=cfg.DATASET.TRAIN_FILE_LIST,
        shuffle=True,
        mode=ModelPhase.TRAIN,
        data_dir=cfg.DATASET.DATA_DIR)

    conf_mat = metrics.ConfusionMatrix(cfg.DATASET.NUM_CLASSES, streaming=False)
    i = 0
    for img, grt, ignore, imgssrc in dataset.generator():

        conf_mat.calculate(grt[np.newaxis,:, : ,np.newaxis], grt[np.newaxis,:, : ,np.newaxis], ignore[np.newaxis,:, :, np.newaxis])
        _, iou = conf_mat.mean_iou()
        _, acc = conf_mat.accuracy()
        print(iou,acc)
        if i > 20:
            break
        i+=1

#%%
ignore.shape
#%%

if __name__ == '__main1__':

    args = parse_args()

    if args.cfg_file is not None:
        cfg.update_from_file("testunet.yaml")
    if args.opts:
        cfg.update_from_list(args.opts)
    if args.enable_ce:
        random.seed(0)
        np.random.seed(0)

    cfg.TRAINER_ID = int(os.getenv("PADDLE_TRAINER_ID", 0))
    cfg.NUM_TRAINERS = int(os.environ.get('PADDLE_TRAINERS_NUM', 1))

    cfg.check_and_infer()
    # print(pprint.pformat(cfg))

    dataset = myDataset.SegDataset(
        file_list=cfg.DATASET.TRAIN_FILE_LIST,
        shuffle=True,
        mode=ModelPhase.TRAIN,
        data_dir=cfg.DATASET.DATA_DIR)

    import os
    import time
    import shutil
    import numpy as np
    import tensorflow as tf
    from tensorflow.keras import backend as K
    from tensorflow import keras

    model = myHRnet.hrnet_keras(num_classes=8)
    model.load_weights("oldModel3.h5")
    # keras.utils.plot_model(model, show_shapes=True, show_layer_names=True)

    tfDataset = tf.data.Dataset.from_generator(dataset.generator, (tf.float32, tf.int32, tf.int32, tf.uint8))
    tfDataset = tfDataset.repeat().batch(12)
    optimizer = tf.keras.optimizers.Adam(0.0002)

    global_steps = 0
    for epoch in range(100):
        i = 0
        for img, grt, ignore, imgssrc in tfDataset:
            with tf.GradientTape() as tape:
                y_pred = model(img, training=True)

                y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())

                lbs = tf.one_hot(grt, 8)
                CE_loss = - lbs * K.log(y_pred)
                CE_loss = K.mean(K.sum(CE_loss, axis=-1))
                tf.print(CE_loss, i)

                # tf.print(y_pred.shape) # 0.125
                # tf.print(K.log(y_pred).shape) # -2.07
                # tf.print((- lbs * K.log(y_pred)).shape)
                print((y_pred)[0,0,0])
                #
                # tf.print(K.sum(- lbs * K.log(y_pred), axis=-1).shape)
                # tf.print(K.sum(- lbs * K.log(y_pred), axis=-1))
                #
                # exit()

                gradients = tape.gradient(CE_loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                if i > 300:
                    break
                i += 1
        global_steps += 1
        tf.print(global_steps)
        model.save("oldModel4.h5")
        tf.print("saveModel")




# %%
