#%%
import os
import cv2
import random
import numpy as np
from config import cfg
from cv2 import cv2 
import glob
import xml.etree.ElementTree as ET

class NpDataset(object):
    def __init__(self, dataset_type):
        self.annot_path  = cfg.TRAIN.ANNOT_PATH if dataset_type == 'train' else cfg.TEST.ANNOT_PATH
        self.no_annot_path = cfg.TRAIN.NoANNOT_PATH if dataset_type == 'train' else cfg.TEST.ANNOT_PATH
        self.input_sizes = cfg.TRAIN.INPUT_SIZE if dataset_type == 'train' else cfg.TEST.INPUT_SIZE
        self.batch_size  = cfg.TRAIN.BATCH_SIZE if dataset_type == 'train' else cfg.TEST.BATCH_SIZE
        self.data_aug    = cfg.TRAIN.DATA_AUG   if dataset_type == 'train' else cfg.TEST.DATA_AUG

        self.train_input_size = cfg.TRAIN.INPUT_SIZE if dataset_type == 'train' else cfg.TEST.INPUT_SIZE

        self.annotations = []
        for path in self.annot_path:
            self.annotations += self.parse_annotations(path)
        for path in self.no_annot_path:
            self.annotations += self.parse_Noannotations(path)

        self.num_samples = len(self.annotations)

        # self.num_batchs = int(np.ceil(self.num_samples / self.batch_size))
        self.num_batchs = cfg.TRAIN.BATCH_NUMS
        self.batch_count = 0

    def parse_Noannotations(self, dir):
        imgs_info = []
        for name in os.listdir(dir):
            img_info = dict()

            imgPath = os.path.join(dir, name)
            print(imgPath)
            img = cv2.imread(imgPath)
            if img.shape[0] < self.train_input_size or img.shape[1] < self.train_input_size:
                continue
            img_info = dict()
            img_info['imgsPath'] = imgPath
            img_info['img'] = img
            img_info['imgsObjects'] = None
            imgs_info.append(img_info)

        return imgs_info

    def parse_annotations(self, ann_dir):
        imgs_info = []
        max_boxes = 0
        for anno in os.listdir(ann_dir):
            tree = ET.parse(os.path.join(ann_dir, anno))
            objects = []
            lables = []
            boxes_counter = 0
            imgP = ''
            img = None
            mask = None
            for elem in tree.iter():
                if 'path' == elem.tag:
                    imgP = ann_dir+'/../' + elem.text[elem.text.rfind('\\')+1:]
                    img = cv2.imread(imgP)
                    print(img.shape, imgP)
                    mask = np.zeros([img.shape[0], img.shape[1]])

                if 'item' == elem.tag:
                    # print(imgP)
                    # lable = 1 # 3 - int(elem.find('name').text)
                    if int(elem.find('name').text) == 1:
                        lable = 1
                    else:
                        continue
                    contours = np.zeros((int(len(elem.find('polygon')) / 2), 2), dtype=np.int)
                    for i,pos in enumerate(list(elem.find('polygon'))):
                        contours[int(i/2), (i%2)] = int(pos.text)


                    cv2.fillPoly(mask, pts=[contours], color=(lable))  # 内部为1
                    cv2.polylines(mask, pts=[contours], isClosed=True, color=lable*0.5, thickness=2) # 边界为0.5

                    objects.append(contours)
                    lables.append(lable)
                    boxes_counter += 1

            if len(objects) == 0:  # 如果没有标注 就跳过这张图片
                continue

            if boxes_counter > max_boxes:
                max_boxes = boxes_counter
            img_info = dict()
            img_info['imgsPath'] = imgP
            img_info['imgsObjects'] = objects
            img_info['imgsLables'] = lables
            img_info['img'] = img
            img_info['mask'] = mask
            imgs_info.append(img_info)
        return imgs_info

    def __len__(self):
        return self.num_batchs


    def __iter__(self):
        self.batch_count = 0
        random.shuffle(self.annotations)  # 改变顺序
        return self  


    def __next__(self):
        if self.batch_count < self.num_batchs:
            batch_image = np.zeros([self.batch_size, self.train_input_size, self.train_input_size, 3], dtype=np.float32)
            batch_labele = np.zeros([self.batch_size, self.train_input_size, self.train_input_size], dtype=np.int32)
            for num in range(self.batch_size):
                index = self.batch_count * self.batch_size + num
                index %= self.num_samples
                annotation = self.annotations[index]
                image, lable = self.parse_annotation(annotation)
                # 在这里可以做一些数据增强
                # tem = np.random.randint(0, 5)
                # if tem == 0:
                #     batch_image[num, :, :, :] = image[:,:,[1,2,0]]
                # elif tem == 1:
                #     batch_image[num, :, :, :] = image[:,:,[2,0,1]]
                # else:
                #     batch_image[num, :, :, :] = image
                batch_image[num, :, :, :] = image
                batch_labele[num, :, :] = lable
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


    def parse_annotation(self, img_info):
        objs = img_info['imgsObjects']

        if np.random.randint(0, 2) == 0 and objs is not None and len(objs) != 0:
            b = 10
            n = np.random.randint(0, len(objs))
            x, y, w, h = cv2.boundingRect(objs[n])
            tx = np.random.randint(x - (self.train_input_size - b), x + w - b)
            ty = np.random.randint(y - (self.train_input_size - b), y + h - b)
            tx = min(max(tx, 0), img_info['img'].shape[1] - self.train_input_size)
            ty = min(max(ty, 0), img_info['img'].shape[0] - self.train_input_size)
            # print(x, y, w, h, tx, ty)
        else:
            tx = np.random.randint(0, img_info['img'].shape[1] - self.train_input_size)
            ty = np.random.randint(0, img_info['img'].shape[0] - self.train_input_size)

        tx = int(tx)
        ty = int(ty)
        imageSrc = img_info['img']
        image = imageSrc[ty:ty+self.train_input_size, tx:tx+self.train_input_size]

        if objs is not None and len(objs) != 0:
            lable = img_info['mask'][ty:ty+self.train_input_size, tx:tx+self.train_input_size]
        else:
            lable = 0
        return image, lable

# if __name__ == '__main__':
#     train_dateset = NpDataset('train')
#     for imgs, lables in train_dateset:
#         # print(lables[3])
#         # print(imgs.shape)
#         # print(lables.shape)
#         # print(np.unique(imgs[0]))
#         cv2.imshow('1', imgs[3]/255.0)
#         cv2.imshow('2', lables[3] * 0.5)
#         cv2.waitKey(0)

if __name__ == '__main__':
    import tensorflow as tf
    train_dateset = NpDataset('train')
    for imgs, lables in train_dateset:
        # print(lables[3])
        # print(imgs.shape)
        # print(lables.shape)
        # print(np.unique(imgs[0]))

        y_onehot = tf.one_hot(lables, depth=3)
        print(y_onehot.shape)
        cv2.imshow('y_onehot', y_onehot[3].numpy()*1.0)
        cv2.imshow('1', imgs[3] / 255.0)
        cv2.imshow('2', lables[3] * 0.5)
        cv2.waitKey(0)


