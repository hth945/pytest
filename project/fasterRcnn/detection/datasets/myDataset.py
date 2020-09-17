from detection.datasets import transforms, utils
import os.path as osp
import cv2
import numpy as np
from cv2 import cv2


class myDataSet(object):

    def __init__(self,  subset='train',
                    flip_ratio=0.0,
                    pad_mode='fixed',
                    mean=(0, 0, 0),
                    std=(1, 1, 1),
                    scale=(1024, 800),
                    debug=False,
                    rootPath=None,
                 ):

        self.flip_ratio = flip_ratio    

        if subset not in ['train', 'val']:
            raise AssertionError('subset must be "train" or "val".')

        self.imgs = []
        max_boxes = 0
        if rootPath==None:
            rootPath = '../../dataAndModel/data/mnist/'
        label_txt = rootPath + 'objtrainlab.txt'
        with open(label_txt) as f:
            while True:
                line = f.readline().split()
                if len(line) == 0:
                    break
                self.imgs.append(line[0])
                if len(line)-1 > max_boxes:
                    max_boxes  = len(line)-1

        self.boxes = np.zeros([len(self.imgs), max_boxes, 4],dtype=np.int)
        self.lables = np.zeros([len(self.imgs), max_boxes],dtype=np.int)
        with open(label_txt) as f:
            index = 0
            while True:
                line = f.readline().split()
                if len(line) == 0:
                    break
            
                for i, bbox in enumerate(line[1:]):
                    # print(bbox)
                    bbox = bbox.split(",")
                    self.boxes[index,i,0] = int(float(bbox[0]))
                    self.boxes[index,i,1] = int(float(bbox[1]))
                    self.boxes[index,i,2] = int(float(bbox[2]))
                    self.boxes[index,i,3] = int(float(bbox[3]))

                    self.lables[index,i] = int(float(bbox[4]))+1
                index += 1

        # self.boxes = []
        # self.lables = []
        # with open(label_txt) as f:
        #     index = 0
        #     while True:
        #         line = f.readline().split()
        #         if len(line) == 0:
        #             break
        #         bboxs = []
        #         blables = []
        #         for i, bbox in enumerate(line[1:]):
        #             # print(bbox)
        #             bbox = bbox.split(",")
        #             bboxs.append([float(bbox[0]),float(bbox[1]),float(bbox[2]),float(bbox[3])])
        #
        #             blables.append(int(float(bbox[4]))+1)
        #         index += 1
        #         self.lables.append(np.array(blables, dtype=np.int64))
        #         self.boxes.append(np.array(bboxs, dtype=np.float32))

        if pad_mode in ['fixed', 'non-fixed']:
            self.pad_mode = pad_mode
        elif subset == 'train':
            self.pad_mode = 'fixed'
        else:
            self.pad_mode = 'non-fixed'

        self.img_transform = transforms.ImageTransform(scale, mean, std, pad_mode)
        self.bbox_transform = transforms.BboxTransform()

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        imgPath = self.imgs[idx]
        box = self.boxes[idx]
        lable = self.lables[idx]

        img = cv2.imread(imgPath, cv2.IMREAD_COLOR).astype(np.float32)
        ori_shape = img.shape

        # flip = True if np.random.rand() < self.flip_ratio else False
        flip = False
        img, img_shape, scale_factor = self.img_transform(img, flip)
        img=1- img/255.0
        pad_shape = img.shape
        box, lable = self.bbox_transform(box, lable, img_shape, scale_factor, flip)

        img_meta_dict = dict({
            'ori_shape': ori_shape,
            'img_shape': img_shape,
            'pad_shape': pad_shape,
            'scale_factor': scale_factor,
            'flip': flip
        })
        img_meta = utils.compose_image_meta(img_meta_dict)
        # print('idx: ', idx)
        # print(img.shape)
        # print(img_meta.shape)
        # print(box.shape)
        # print(lable.shape)

        # return img, img, img, img
        return img, img_meta, box, lable
    def get_categories(self):
        return [0,1,2,3,4,5,6,7,8,9,10]
#%%
if __name__ == '__main1__':
    train_dataset = myDataSet(flip_ratio=0.5, scale=(768, 768), rootPath='../../../../dataAndModel/data/mnist/')

    # img = cv2.imread(train_dataset.imgs[0], cv2.IMREAD_COLOR)
    # cv2.imshow('img', img)
    # img2, img_shape, scale_factor = train_dataset.img_transform(img, True)
    # img2 = img2/255.0
    # cv2.imshow('img2', img2)
    # print(img,img2)
    # cv2.waitKey(0)

    for img, img_meta, box, lable in train_dataset:
        cv2.imshow('img', img)
        cv2.waitKey(0)

# %%

if __name__ == '__main__':
    import tensorflow as tf
    import data_generator
    train_dataset = myDataSet(flip_ratio=0.5, scale=(768, 768), rootPath='../../../../dataAndModel/data/mnist/')


    num_classes = len(train_dataset.get_categories())
    train_generator = data_generator.DataGenerator(train_dataset)
    train_tf_dataset = tf.data.Dataset.from_generator(train_generator, (tf.float32, tf.float32, tf.float32, tf.int32))

    train_tf_dataset = train_tf_dataset.batch(2)

    for inputs in train_tf_dataset:
        for i in inputs:
            print(i.shape)
        break