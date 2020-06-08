#%%
# 
import cv2
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from tensorflow import keras
import tensorflow.keras.backend as K
from myconfig import cfg
import os,glob
from matplotlib import pyplot as plt
from matplotlib import patches
from myDataset import *
from myconfig import cfg
from yoloNet import *

# 5.2
def visualize_result(img, model):
    IMGSZ = cfg.TRAIN.IMGSZ
    GRIDSZ = cfg.TRAIN.GRIDSZ
    ANCHORS = cfg.TRAIN.ANCHORS

    # [512,512,3] 0~255, BGR
    img = cv2.imread(img)
    img = img[...,::-1]/255.
    img = tf.cast(img, dtype=tf.float32)
    # [1,512,512,3]
    img = tf.expand_dims(img, axis=0)
    # [1,16,16,5,7]
    y_pred = model(img, training=False)

    x_grid = tf.tile(tf.range(GRIDSZ), [GRIDSZ])
    # [1, 16,16,1,1]
    x_grid = tf.reshape(x_grid, (1, GRIDSZ, GRIDSZ, 1, 1))
    x_grid = tf.cast(x_grid, dtype=tf.float32)
    y_grid = tf.transpose(x_grid, (0,2,1,3,4))
    xy_grid = tf.concat([x_grid,y_grid], axis=-1)
    # [1, 16, 16, 5, 2]
    xy_grid = tf.tile(xy_grid, [1, 1, 1, 5, 1])

    anchors = np.array(ANCHORS).reshape(5,2)
    pred_xy = tf.sigmoid(y_pred[...,0:2])
    pred_xy = pred_xy + xy_grid
    # normalize 0~1
    pred_xy = pred_xy / tf.constant([16.,16.])

    pred_wh = tf.exp(y_pred[...,2:4])
    pred_wh = pred_wh * anchors
    pred_wh = pred_wh / tf.constant([16.,16.])

    # [1,16,16,5,1]
    pred_conf = tf.sigmoid(y_pred[...,4:5])
    # l1 l2
    pred_prob = tf.nn.softmax(y_pred[...,5:])

    pred_xy, pred_wh, pred_conf, pred_prob = \
        pred_xy[0], pred_wh[0], pred_conf[0], pred_prob[0]

    boxes_xymin = pred_xy - 0.5 * pred_wh
    boxes_xymax = pred_xy + 0.5 * pred_wh
    # [16,16,5,2+2]
    boxes = tf.concat((boxes_xymin, boxes_xymax),axis=-1)
    # [16,16,5,2]
    box_score = pred_conf * pred_prob
    # [16,16,5]
    box_class = tf.argmax(box_score, axis=-1)
    # [16,16,5]
    box_class_score = tf.reduce_max(box_score, axis=-1)
    # [16,16,5]
    pred_mask = box_class_score > 0.45
    # [16,16,5,4]=> [N,4]
    boxes = tf.boolean_mask(boxes, pred_mask)
    # [16,16,5] => [N]
    scores = tf.boolean_mask(box_class_score, pred_mask)
    # 【16,16，5】=> [N]
    classes = tf.boolean_mask(box_class, pred_mask)

    boxes = boxes * 512.
    # [N] => [n]
    select_idx = tf.image.non_max_suppression(boxes, scores, 40, iou_threshold=0.3)
    boxes = tf.gather(boxes, select_idx)
    scores = tf.gather(scores, select_idx)
    classes = tf.gather(classes, select_idx)

    # plot
    fig, ax = plt.subplots(1, figsize=(10,10))
    ax.imshow(img[0])
    n_boxes = boxes.shape[0]
    ax.set_title('boxes:%d'%n_boxes)
    for i in range(n_boxes):
        x1,y1,x2,y2 = boxes[i]
        w = x2 - x1
        h = y2 - y1
        label = classes[i].numpy()

        if label==0: # sugarweet
            color = (0,1,0)
        else:
            color = (1,0,0)

        rect = patches.Rectangle((x1.numpy(), y1.numpy()), w.numpy(), h.numpy(), linewidth = 3, edgecolor=color,facecolor='none')
        ax.add_patch(rect)


model = yoloNetModle()
model.load_weights('weights/epoch10.ckpt')
#%%
files = glob.glob('data/val/image/*.png')
for x in files:
    visualize_result(x, model)
plt.show()