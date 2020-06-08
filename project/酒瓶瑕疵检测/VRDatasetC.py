#%%
import cv2
import numpy as np
from config import cfg
from cv2 import cv2
import json
import random

# category_dic= {1: '瓶盖破损',
#     9: '喷码正常',
#     5: '瓶盖断点',
#     3: '瓶盖坏边',
#     4: '瓶盖打旋',
#     0: '背景',
#     2: '瓶盖变形',
#     8: '标贴气泡',
#     6: '标贴歪斜',
#     10: '喷码异常',
#     7: '标贴起皱'}

class NpDataset(object):
    def __init__(self, dataset_type):
        self.annot_path  = cfg.TRAIN.ANNOT_PATH if dataset_type == 'train' else cfg.TEST.ANNOT_PATH
        self.input_sizes = cfg.TRAIN.INPUT_SIZE if dataset_type == 'train' else cfg.TEST.INPUT_SIZE
        self.batch_size  = cfg.TRAIN.BATCH_SIZE if dataset_type == 'train' else cfg.TEST.BATCH_SIZE

        self.annotations = self.parse_annotations(self.annot_path)

        self.num_samples = len(self.annotations)
        self.num_batchs = cfg.TRAIN.BATCH_NUMS
        self.batch_count = 0

    def parse_annotations(self,path):

        with open(path+'/annotations.json') as f:
            a = json.load(f)

        data = []
        annotas = a['annotations']
        for img in a['images']:
            if img['height'] != 492:
                continue
            sample_img = img
            sample_annota_list = []
            for per in annotas:
                if img['id'] == per['image_id'] and per['category_id'] != 0 and per['category_id'] <= 5:
                    sample_annota_list.append(per)
            for k in sample_annota_list:
                annotas.remove(k)
            if len(sample_annota_list) > 0:
                sample_img['annotations'] = sample_annota_list
                data.append(sample_img)

        imgsinfo = []
        for img_data in data:
            img = cv2.imread(path+'/images/' + img_data['file_name'])
            imginfo = {}
            imginfo['img'] = img
            img_annotations = img_data['annotations']
            n = len(img_annotations)
            boxs = np.zeros((n, 5), dtype=np.int32)
            for i in range(n):
                bbox = img_annotations[i]['bbox']
                boxs[i, 0] = img_annotations[i]['category_id']

                x1 = max(0, np.floor(bbox[0] + 0.5).astype('int32'))
                y1 = max(0, np.floor(bbox[1] + 0.5).astype('int32'))
                x2 = min(img.shape[1], np.floor(bbox[2] + bbox[0] + 0.5).astype('int32'))
                y2 = min(img.shape[0], np.floor(bbox[3] + bbox[1] + 0.5).astype('int32'))
                # boxs[i, 1:5] = [x1, y1, x2, y2]
                boxs[i, 1:5] = [y1, x1, y2, x2]
            imginfo['img_info'] = img_data
            imginfo['boxs'] = boxs
            imgsinfo.append(imginfo)
            if len(imgsinfo) > 20:
                break
        return imgsinfo

    def __len__(self):
        return self.num_batchs

    def __iter__(self):
        self.batch_count = 0
        random.shuffle(self.annotations)
        # random.shuffle(self.annotations)  # 改变顺序
        return self  

    def __next__(self):
        if self.batch_count < self.num_batchs:
            batch_image = np.zeros([self.batch_size, self.input_sizes[0], self.input_sizes[1],3], dtype=np.uint8)
            batch_labele = np.zeros([self.batch_size, self.input_sizes[0], self.input_sizes[1]], dtype=np.uint8)
            for num in range(self.batch_size):
                index = self.batch_count * self.batch_size + num
                index %= self.num_samples
                annotation = self.annotations[index]
                self.parse_annotation(annotation, batch_image[num, :, :,:], batch_labele[num, :, :])

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
        boxs = img_info['boxs']
        img = img_info['img']
        # print(img_info)
        masks = np.zeros([img.shape[0], img.shape[1]],dtype=np.uint8)

        for i in range(boxs.shape[0]):
            for j in range(10):
                cv2.rectangle(masks, (boxs[i, 2], boxs[i, 1]), (boxs[i, 4], boxs[i, 3]),  (20*j+50), 10-j, 4)
        for i in range(boxs.shape[0]):
            masks[boxs[i, 1]:boxs[i, 3], boxs[i, 2]:boxs[i, 4]] = 255
        # print(batchImg.shape)
        # print(img.shape)
        # [492,658] # 512, 640
        batchImg[:img.shape[0],:,:] = img[:,:640]
        batchLable[:img.shape[0],:] = masks[:,:640]


if __name__ == '__main2__':
    train_dateset = NpDataset('train')
    # train_dateset = NpDataset('test')
    for imgs, lables in train_dateset:
        # print(lables[3])
        print(imgs.shape)
        print(lables.shape)
        # print(np.unique(imgs[0]))
        cv2.imshow('1', imgs[3]/255.0)
        cv2.imshow('2', lables[3] /255.0)
        cv2.waitKey(0)

if __name__ == '__main__':
    import tensorflow as tf

    modelTest = tf.keras.models.load_model('oldModel.h5')
    train_dateset = NpDataset('train')
    # train_dateset = NpDataset('test')
    for imgs, lables in train_dateset:
        img,lab = modelTest(imgs/255.0)

        cv2.imshow('imgsSrc', imgs[3] / 255.0)
        cv2.imshow('lab', lab[3].numpy())
        cv2.imshow('lablesSrc', lables[3] / 255)

        # blank = lab[3].numpy()
        # labTem = np.zeros([cfg.TRAIN.INPUT_SIZE,cfg.TRAIN.INPUT_SIZE])
        #
        # for i in range(4):
        #     tem = blank[:,:,i]
        #
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




