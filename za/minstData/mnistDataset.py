#%%
import cv2
from cv2 import cv2
import numpy as np
import shutil
import os
import random
from PIL import Image
import matplotlib.pyplot as plt 
import tensorflow as tf 
from tensorflow import keras 

from matplotlib import pyplot as plt 
from matplotlib import patches 

def parse_annotation():
    imgs = []
    max_boxes = 0
    rootPath = '../../dataAndModel/data/mnist/'
    label_txt = rootPath + 'objtrainlab.txt'
    with open(label_txt) as f:
        while True:
            line = f.readline().split()
            if len(line) == 0:
                break
            imgs.append(line[0])
            if len(line)-1 > max_boxes:
                max_boxes  = len(line)-1

    boxes = np.zeros([len(imgs), max_boxes, 5])
    with open(label_txt) as f:
        index = 0
        while True:
            line = f.readline().split()
            if len(line) == 0:
                break
        
            for i, bbox in enumerate(line[1:]):
                # print(bbox)
                bbox = bbox.split(",")
                boxes[index,i,0] = int(float(bbox[0]))
                boxes[index,i,1] = int(float(bbox[1]))
                boxes[index,i,2] = int(float(bbox[2]))
                boxes[index,i,3] = int(float(bbox[3]))
                boxes[index,i,4] = int(float(bbox[4]))+1
            index += 1
    return imgs, boxes

def preprocess(img, img_boxes):
    # img: string
    # img_boxes: [40,5]
    x = tf.io.read_file(img)
    x = tf.io.decode_jpeg(x, channels=3)
    x = tf.image.convert_image_dtype(x, tf.float32)
    return x, img_boxes

def get_dataset():
    imgs, boxes = parse_annotation()
    db = tf.data.Dataset.from_tensor_slices((imgs, boxes))

    db = db.shuffle(1000).map(preprocess).batch(10).repeat()
    print('db Images:', len(imgs))
    return db


def db_visualize(db):
    # imgs:[b, 512, 512, 3]
    # imgs_boxes: [b, 40, 5]
    imgs, imgs_boxes = next(iter(db))
    img, img_boxes = imgs[0], imgs_boxes[0]
    f,ax1 = plt.subplots(1,figsize=(10,10))
    # display the image, [512,512,3]
    ax1.imshow(img)
    for x1,y1,x2,y2,l in img_boxes: # [40,5]
        if l == 0:
            break
        x1,y1,x2,y2 = float(x1), float(y1), float(x2), float(y2)
        w = x2 - x1 
        h = y2 - y1 

        # if l==1: # green for sugarweet
        #     color = (0,1,0)
        # elif l==2: # red for weed
        #     color = (1,0,0) # (R,G,B)
        # else: # ignore invalid boxes
        #     break
        color = (1,0,0)
        rect = patches.Rectangle((x1,y1), w, h, linewidth=2,edgecolor=color, facecolor='none')
        ax1.add_patch(rect)
        caption = "{:.0f} ".format(l-1)
        ax1.text(x1, y1, caption,color=(0,0,0), size=11, backgroundcolor="none")

#%%
train_db = get_dataset()
db_visualize(train_db)
