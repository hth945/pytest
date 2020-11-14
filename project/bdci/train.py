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


def pil_imread(file_path):
    """read pseudo-color label"""
    im = Image.open(file_path)
    return np.asarray(im)


def cv2_imread(file_path, flag=cv2.IMREAD_COLOR):
    # resolve cv2.imread open Chinese file path issues on Windows Platform.
    return cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), flag)


class SegDataset(object):
    def __init__(self,
                 file_list,
                 data_dir,
                 mode=ModelPhase.TRAIN,
                 shuffle=False, ):
        self.shuffle = shuffle
        self.data_dir = data_dir
        self.mode = mode

        self.shuffle_seed = 0
        # NOTE: Please ensure file list was save in UTF-8 coding format
        with codecs.open(file_list, 'r', 'utf-8') as flist:
            self.lines = [line.strip() for line in flist]
            self.all_lines = copy.deepcopy(self.lines)
            if shuffle and cfg.NUM_TRAINERS > 1:
                np.random.RandomState(self.shuffle_seed).shuffle(self.all_lines)
            elif shuffle:
                np.random.shuffle(self.lines)

    def generator(self):
        if self.shuffle and cfg.NUM_TRAINERS > 1:
            np.random.RandomState(self.shuffle_seed).shuffle(self.all_lines)
            num_lines = len(self.all_lines) // cfg.NUM_TRAINERS
            self.lines = self.all_lines[num_lines * cfg.TRAINER_ID:num_lines *
                                                                   (cfg.TRAINER_ID + 1)]
            self.shuffle_seed += 1
        elif self.shuffle:
            np.random.shuffle(self.lines)

        for line in self.lines:
            yield self.process_image(line, self.data_dir, self.mode)

    def process_image(self, line, data_dir, mode):
        """ process_image """
        img, grt, img_name, grt_name = self.load_image(
            line, data_dir, mode=mode)

        imgssrc = img.copy()
        if mode == ModelPhase.TRAIN:
            img, grt = aug.resize(img, grt, mode)
            if cfg.AUG.RICH_CROP.ENABLE:
                if cfg.AUG.RICH_CROP.BLUR:
                    if cfg.AUG.RICH_CROP.BLUR_RATIO <= 0:
                        n = 0
                    elif cfg.AUG.RICH_CROP.BLUR_RATIO >= 1:
                        n = 1
                    else:
                        n = int(1.0 / cfg.AUG.RICH_CROP.BLUR_RATIO)
                    if n > 0:
                        if np.random.randint(0, n) == 0:
                            radius = np.random.randint(3, 10)
                            if radius % 2 != 1:
                                radius = radius + 1
                            if radius > 9:
                                radius = 9
                            img = cv2.GaussianBlur(img, (radius, radius), 0, 0)

                img, grt = aug.random_rotation(
                    img,
                    grt,
                    rich_crop_max_rotation=cfg.AUG.RICH_CROP.MAX_ROTATION,
                    mean_value=cfg.DATASET.PADDING_VALUE)

                img, grt = aug.rand_scale_aspect(
                    img,
                    grt,
                    rich_crop_min_scale=cfg.AUG.RICH_CROP.MIN_AREA_RATIO,
                    rich_crop_aspect_ratio=cfg.AUG.RICH_CROP.ASPECT_RATIO)
                img = aug.hsv_color_jitter(
                    img,
                    brightness_jitter_ratio=cfg.AUG.RICH_CROP.
                        BRIGHTNESS_JITTER_RATIO,
                    saturation_jitter_ratio=cfg.AUG.RICH_CROP.
                        SATURATION_JITTER_RATIO,
                    contrast_jitter_ratio=cfg.AUG.RICH_CROP.
                        CONTRAST_JITTER_RATIO)

            if cfg.AUG.FLIP:
                if cfg.AUG.FLIP_RATIO <= 0:
                    n = 0
                elif cfg.AUG.FLIP_RATIO >= 1:
                    n = 1
                else:
                    n = int(1.0 / cfg.AUG.FLIP_RATIO)
                if n > 0:
                    if np.random.randint(0, n) == 0:
                        img = img[::-1, :, :]
                        grt = grt[::-1, :]

            if cfg.AUG.MIRROR:
                if np.random.randint(0, 2) == 1:
                    img = img[:, ::-1, :]
                    grt = grt[:, ::-1]

            img, grt = aug.rand_crop(img, grt, mode=mode)
        elif ModelPhase.is_eval(mode):
            img, grt = aug.resize(img, grt, mode=mode)
            img, grt = aug.rand_crop(img, grt, mode=mode)
        elif ModelPhase.is_visual(mode):
            org_shape = [img.shape[0], img.shape[1]]
            img, grt = aug.resize(img, grt, mode=mode)
            valid_shape = [img.shape[0], img.shape[1]]
            img, grt = aug.rand_crop(img, grt, mode=mode)
        else:
            raise ValueError("Dataset mode={} Error!".format(mode))

        # Normalize image
        if cfg.AUG.TO_RGB:
            img = img[..., ::-1]
        img = self.normalize_image(img)

        if ModelPhase.is_train(mode) or ModelPhase.is_eval(mode):
            # grt = np.expand_dims(np.array(grt).astype('int32'), axis=2)
            grt = np.array(grt).astype('int32')
            grt[grt == cfg.DATASET.IGNORE_INDEX] = 7
            ignore = (grt != cfg.DATASET.IGNORE_INDEX).astype('int32')

        if ModelPhase.is_train(mode):
            return (img, grt, ignore, imgssrc)
        elif ModelPhase.is_eval(mode):
            return (img, grt, ignore)
        elif ModelPhase.is_visual(mode):
            return (img, grt, img_name, valid_shape, org_shape)

    def load_image(self, line, src_dir, mode=ModelPhase.TRAIN):
        # original image cv2.imread flag setting
        cv2_imread_flag = cv2.IMREAD_COLOR
        if cfg.DATASET.IMAGE_TYPE == "rgba":
            # If use RBGA 4 channel ImageType, use IMREAD_UNCHANGED flags to
            # reserver alpha channel
            cv2_imread_flag = cv2.IMREAD_UNCHANGED

        parts = line.strip().split(cfg.DATASET.SEPARATOR)
        if len(parts) != 2:
            if mode == ModelPhase.TRAIN or mode == ModelPhase.EVAL:
                raise Exception("File list format incorrect! It should be"
                                " image_name{}label_name\\n".format(
                    cfg.DATASET.SEPARATOR))
            img_name, grt_name = parts[0], None
        else:
            img_name, grt_name = parts[0], parts[1]

        img_path = os.path.join(src_dir, img_name)
        img = cv2_imread(img_path, cv2_imread_flag)

        if grt_name is not None:
            grt_path = os.path.join(src_dir, grt_name)
            grt = pil_imread(grt_path)
        else:
            grt = None

        if img is None:
            raise Exception(
                "Empty image, source image path: {}".format(img_path))

        img_height = img.shape[0]
        img_width = img.shape[1]

        if grt is not None:
            grt_height = grt.shape[0]
            grt_width = grt.shape[1]

            if img_height != grt_height or img_width != grt_width:
                raise Exception(
                    "Source img and label img must has the same size.")
        else:
            if mode == ModelPhase.TRAIN or mode == ModelPhase.EVAL:
                raise Exception(
                    "No laber image path for image '{}' when training or evaluating. "
                        .format(img_path))

        if len(img.shape) < 3:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        img_channels = img.shape[2]
        if img_channels < 3:
            raise Exception("PaddleSeg only supports gray, rgb or rgba image")
        if img_channels != cfg.DATASET.DATA_DIM:
            raise Exception(
                "Input image channel({}) is not match cfg.DATASET.DATA_DIM({}), img_name={}"
                    .format(img_channels, cfg.DATASET.DATADIM, img_name))
        if img_channels != len(cfg.MEAN):
            raise Exception(
                "Image name {}, image channels {} do not equal the length of cfg.MEAN {}."
                    .format(img_name, img_channels, len(cfg.MEAN)))
        if img_channels != len(cfg.STD):
            raise Exception(
                "Image name {}, image channels {} do not equal the length of cfg.STD {}."
                    .format(img_name, img_channels, len(cfg.STD)))

        return img, grt, img_name, grt_name

    def normalize_image(self, img):
        """ 像素归一化后减均值除方差 """
        # img = img.transpose((2, 0, 1)).astype('float32') / 255.0
        # img_mean = np.array(cfg.MEAN).reshape((len(cfg.MEAN), 1, 1))
        # img_std = np.array(cfg.STD).reshape((len(cfg.STD), 1, 1))

        img = img.astype('float32') / 255.0
        img_mean = np.array(cfg.MEAN).reshape((1, 1, len(cfg.MEAN)))
        img_std = np.array(cfg.STD).reshape((1, 1, len(cfg.STD)))

        img -= img_mean
        img /= img_std

        return img


if __name__ == '__main__':

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

    dataset = SegDataset(
        file_list=cfg.DATASET.TRAIN_FILE_LIST,
        shuffle=True,
        mode=ModelPhase.TRAIN,
        data_dir=cfg.DATASET.DATA_DIR)

    # data_gen = dataset.generator()
    # i = 0
    # for b in data_gen:
    # img, grt, ignore,imgssrc = b

    # # print(img.shape)
    # # print(grt.shape)
    # # print(ignore.shape)
    # # print(imgssrc.shape)

    # grt = list(set(grt.flatten()))
    # print("grt:", grt)
    # ignore = list(set(ignore.flatten()))
    # print("ignore:", ignore)
    # if i > 2:
    # break
    # i += 1

    # import tensorflow as tf
    # tfDataset = tf.data.Dataset.from_generator(dataset.generator, (tf.float32,tf.int32,tf.int32,tf.uint8)).batch(2)
    # for i in tfDataset:
    # print(i)
    # break

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



