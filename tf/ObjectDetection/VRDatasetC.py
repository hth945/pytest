#%%
import os
import cv2
import random
import numpy as np
from config import cfg
from cv2 import cv2 
import glob
import xml.etree.ElementTree as ET
from imgUtil import *


class NpDataset(object):
    def __init__(self, dataset_type):
        self.annot_path  = cfg.TRAIN.ANNOT_PATH if dataset_type == 'train' else cfg.TEST.ANNOT_PATH
        self.no_annot_path = cfg.TRAIN.NoANNOT_PATH if dataset_type == 'train' else cfg.TEST.ANNOT_PATH
        self.input_sizes = cfg.TRAIN.INPUT_SIZE if dataset_type == 'train' else cfg.TEST.INPUT_SIZE
        self.batch_size  = cfg.TRAIN.BATCH_SIZE if dataset_type == 'train' else cfg.TEST.BATCH_SIZE
        self.data_aug    = cfg.TRAIN.DATA_AUG   if dataset_type == 'train' else cfg.TEST.DATA_AUG
        self.train_input_size = cfg.TRAIN.INPUT_SIZE if dataset_type == 'train' else cfg.TEST.INPUT_SIZE

        self.imgSrc = cv2.imread("imgSrc.bmp", cv2.IMREAD_GRAYSCALE)
        self.pointsSrc = np.array([[1, 6], [1, 22], [9, 6], [9, 22]])
        x, y = np.mgrid[0:360:1, 0.5:1:0.1]
        self.annotations = np.c_[x.ravel(), y.ravel()]

        self.num_samples = self.annotations.shape[0]
        self.num_batchs = cfg.TRAIN.BATCH_NUMS
        self.batch_count = 0

    def __len__(self):
        return self.num_batchs

    def __iter__(self):
        self.batch_count = 0
        np.random.shuffle(self.annotations)
        # random.shuffle(self.annotations)  # 改变顺序
        return self  

    def __next__(self):
        if self.batch_count < self.num_batchs:
            batch_image = np.zeros([self.batch_size, self.train_input_size, self.train_input_size], dtype=np.float32)
            batch_labele = np.zeros([self.batch_size, self.train_input_size, self.train_input_size, cfg.POINTS_NUMBER], dtype=np.int32)
            for num in range(self.batch_size):
                index = self.batch_count * self.batch_size + num
                index %= self.num_samples
                annotation = self.annotations[index]
                self.parse_annotation(annotation, batch_image[num, :, :], batch_labele[num, :, :, :])

            self.batch_count += 1
            return batch_image, batch_labele
        else:
            raise StopIteration

    def bbox_area_check(self, objs, boxes):
        for obj in objs:
            x1, y1, w1, h1 = boxes
            x2, y2, w2, h2 = cv2.boundingRect(obj)
            iW = w1 + w2 - (max(x1+w1, x2+w2) - min(x1, x2))
            iH = h1 + h2 - (max(y1+h1, y2+h2) - min(y1, y2))
            if iW > 1 and iH > 1:
                return 1
        return 0


    def parse_annotation(self, img_info, batchImg, batchLable):
        SIZE = self.input_sizes
        blank = np.zeros(shape=[SIZE, SIZE])
        boxes = []
        pointLabels = []
        for i in range(5):
            # angle = random.randint(0, 360)
            angle = img_info[0]
            img, points = makeSpinImage(self.imgSrc, angle, self.pointsSrc)
            xmin = np.random.randint(0, SIZE - img.shape[0], 1)[0]
            ymin = np.random.randint(0, SIZE - img.shape[1], 1)[0]
            xmax = xmin + img.shape[0]
            ymax = ymin + img.shape[1]
            box = [xmin, ymin, xmax, ymax]
            if len(boxes) > 0:
                iou = [compute_iou(box, b) for b in boxes]
                if max(iou) > 0.02:
                    continue
            points += np.array([xmin, ymin])
            boxes.append(box)
            pointLabels.append(points)
            blank[xmin:xmax, ymin:ymax] = img
        batchImg[:,:] = blank
        for j in range(cfg.POINTS_NUMBER):
            tem = np.zeros(shape=[SIZE, SIZE])
            for points in pointLabels:
                for i in range(5):
                    cv2.circle(tem, (int(points[j, 1]), int(points[j, 0])), i, (255 - i * 40), 1)
            batchLable[:,:,j] = tem




if __name__ == '__main__':
    train_dateset = NpDataset('train')
    for imgs, lables in train_dateset:
        # print(lables[3])
        # print(imgs.shape)
        # print(lables.shape)
        # print(np.unique(imgs[0]))
        cv2.imshow('1', imgs[3]/255.0)
        cv2.imshow('2', lables[3] * 0.5)
        cv2.waitKey(0)

# if __name__ == '__main__':
#     import tensorflow as tf
#     train_dateset = NpDataset('train')
#     for imgs, lables in train_dateset:
#         # print(lables[3])
#         # print(imgs.shape)
#         # print(lables.shape)
#         # print(np.unique(imgs[0]))
#
#         y_onehot = tf.one_hot(lables, depth=3)
#         print(y_onehot.shape)
#         cv2.imshow('y_onehot', y_onehot[3].numpy()*1.0)
#         cv2.imshow('1', imgs[3] / 255.0)
#         cv2.imshow('2', lables[3] * 0.5)
#         cv2.waitKey(0)


