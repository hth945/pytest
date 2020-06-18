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

        simgSrc = cv2.imread("imgSrc.bmp")

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        kimgSrc = cv2.dilate(simgSrc, kernel)  # 膨胀
        kimgSrc[:, :, 1:3] = 0
        kimgSrc[:, :, 0] = kimgSrc[:, :, 0] / 255.0
        simgSrc[:, :, 0:2] = 0
        simgSrc[:, :, 2] = simgSrc[:, :, 2] / 255.0
        a = np.linspace(0, 179, 180)

        self.kimgSrcs = []
        for i in range(a.shape[0]):
            self.kimgSrcs.append(makeSpinImage(kimgSrc, a[i]))

        self.simgSrcs = []
        for i in range(a.shape[0]):
            self.simgSrcs.append(makeSpinImage(simgSrc, a[i]))


        self.num_samples = 2000
        self.num_batchs = cfg.TRAIN.BATCH_NUMS
        self.batch_count = 0

    def __len__(self):
        return self.num_batchs

    def __iter__(self):
        self.batch_count = 0
        # random.shuffle(self.annotations)  # 改变顺序
        return self  

    def __next__(self):
        if self.batch_count < self.num_batchs:
            batch_image = np.zeros([self.batch_size, self.train_input_size, self.train_input_size, 3], dtype=np.float32)
            batch_labele = np.zeros([self.batch_size, 12], dtype=np.float32)
            for num in range(self.batch_size):
                index = self.batch_count * self.batch_size + num
                index %= self.num_samples
                self.parse_annotation( batch_image[num, :, :, :], batch_labele[num, :])

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

    def parse_annotation(self, batchImg, lab):

        # lab = np.array([112, 112, 37, 112, 112, 50], dtype=np.int32)

        angle = np.random.randint(0,len(self.kimgSrcs))
        lab[2] = angle
        img = self.kimgSrcs[angle]
        lab[0] = np.random.randint(img.shape[0]/2+1, batchImg.shape[0] - img.shape[0]/2)
        lab[1] = np.random.randint(img.shape[1]/2+1, batchImg.shape[1] - img.shape[1]/2)
        startx = int(lab[0] - img.shape[0] / 2)
        starty = int(lab[1] - img.shape[1] / 2)
        # print(startx, starty, img.shape)
        batchImg[startx:startx + img.shape[0], starty:starty + img.shape[1], :] += img

        angle = np.random.randint(0, len(self.simgSrcs))
        lab[5] = angle
        img = self.simgSrcs[angle]
        lab[3] = np.random.randint(img.shape[0] / 2 + 1, batchImg.shape[0] - img.shape[0] / 2)
        lab[4] = np.random.randint(img.shape[1] / 2 + 1, batchImg.shape[1] - img.shape[1] / 2)
        startx = int(lab[3] - img.shape[0] / 2)
        starty = int(lab[4] - img.shape[1] / 2)
        # print(startx,starty, img.shape)
        batchImg[startx:startx + img.shape[0], starty:starty + img.shape[1], :] += img

        lab[6:9] = lab[3:6] - lab[0:3]
        lab[9:12][lab[6:9] > 0] = 1


if __name__ == '__main1__':
    train_dateset = NpDataset('train')
    for imgs, lables in train_dateset:
        # print(lables[3])
        print(imgs.shape)
        print(lables.shape)
        # print(np.unique(imgs[0]))
        cv2.imshow('1', imgs[3])
        print(lables[3])
        cv2.waitKey(0)

if __name__ == '__main__':
    import tensorflow as tf
    import tensorflow_hub as hub
    modelTest = tf.keras.models.load_model('oldModel3.h5',custom_objects={'KerasLayer':hub.KerasLayer})
    train_dateset = NpDataset('train')
    for imgs, lables in train_dateset:
        lab = modelTest(imgs[0:4])
        # lab = lab * np.array([224,224,360,224,224,360,22.4,22.4,36.0])
        cv2.imshow('imgsSrc', imgs[3])
        print(lables[3])
        print(lab[3])

        # cv2.imshow('lab', np.sum(lab[3].numpy(), -1))
        # cv2.imshow('lablesSrc', np.sum(lables[3], -1) / 255)
        #
        # blank = lab[3].numpy()
        # labTem = np.zeros([cfg.TRAIN.INPUT_SIZE,cfg.TRAIN.INPUT_SIZE])
        #
        # for i in range(4):
        #     tem = blank[:,:,i]
        #     ret, binary = cv2.threshold(tem, 0.7, 1, cv2.THRESH_BINARY)
        #     labTem += binary
        #     gray = np.uint8(binary * 255.0)
        #     contours, hier = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #     for c in contours:
        #         # area = cv2.contourArea(c)  # 面积
        #         Rectx, Recty, Rectw, Recth = cv2.boundingRect(c)  # 矩形框
        #         imgTem = tem[Recty:Recty + Recth, Rectx:Rectx + Rectw]
        #
        #         # 利用cv2.minMaxLoc寻找到图像中最亮和最暗的点
        #         (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(imgTem)
        #         print(maxLoc)
        #         print(imgTem)
        #
        #     # cv2.imshow('%d'%(i+3), binary)
        # cv2.imshow('labTem', labTem)
        cv2.waitKey(0)




