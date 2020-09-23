#%%
import  os
# import warnings
# warnings.filterwarnings('ignore') # 注：放的位置也会影响效果，真是奇妙的代码
#

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import  numpy as np
from    matplotlib import pyplot as plt
import cv2
from    detection.datasets import myDataset,data_generator
from    detection.models import faster_rcnn2
import  tensorflow as tf
from    tensorflow import keras
from cv2 import cv2
import matplotlib.pyplot as plt 
from detection.core.anchor import anchor_generator, anchor_target
from detection.utils.misc import *
from detection.core.bbox import geometry, transforms



def showRunRpn(batch_imgs, batch_metas,rpn_class_logits, rpn_probs,number=25):
    generator = anchor_generator.AnchorGenerator(
                scales=(32, 64, 128, 256, 512),
                ratios=(0.5, 1, 2),
                feature_strides=(4, 8, 16, 32, 64))
    anchors, valid_flags = generator.generate_pyramid_anchors(batch_metas)
    rpn_probs = rpn_probs[0, :, 1]
    valid_flags = valid_flags[0]
    valid_flags = tf.cast(valid_flags, tf.bool)
    rpn_probs = tf.boolean_mask(rpn_probs, valid_flags)
    anchors = tf.boolean_mask(anchors, valid_flags)
    ix = tf.nn.top_k(rpn_probs, number, sorted=True).indices
    rpn_probs = tf.gather(rpn_probs, ix)
    anchors = tf.gather(anchors, ix)
    image = batch_imgs[0].numpy()
    bboxs = anchors.numpy()
    for i in range(bboxs.shape[0]):
        bbox = bboxs[i]
        image = cv2.rectangle(image, (int(float(bbox[0])),
                                    int(float(bbox[1]))),
                                    (int(float(bbox[2])),
                                    int(float(bbox[3]))), (1.0, 0, 0), 2)
    return image

def showLabRpn(batch_imgs, batch_metas, batch_bboxes, batch_labels):
    generator = anchor_generator.AnchorGenerator(
                scales=(32, 64, 128, 256, 512),
                ratios=(0.5, 1, 2),
                feature_strides=(4, 8, 16, 32, 64))
    anchors, valid_flags = generator.generate_pyramid_anchors(batch_metas)
    # anchors = anchors[0]
    valid_flags = valid_flags[0]

    gt_boxes, _ = trim_zeros(batch_bboxes[0])
    target_matchs = tf.zeros(anchors.shape[0], dtype=tf.int32)
     # Compute overlaps [num_anchors, num_gt_boxes] 326393 vs 10 => [326393, 10]
    overlaps = geometry.compute_overlaps(anchors, gt_boxes)
    anchor_iou_max = tf.reduce_max(overlaps, axis=[1]) # [326396] get closet gt boxes's overlap scores
    target_matchs = tf.where(anchor_iou_max > 0.55, True, False)

    tem = tf.where(anchor_iou_max < 0.01, True, False)
    print(tf.boolean_mask(anchors, tem).shape)
    
    anchors = tf.boolean_mask(anchors, target_matchs)
    print(anchors.shape)
    

    image = batch_imgs[0].numpy()
    bboxs = anchors.numpy()
    for i in range(bboxs.shape[0]):
        bbox = bboxs[i]# *768
        # print(bbox)
        image = cv2.rectangle(image, (int(float(bbox[0])),
                                      int(float(bbox[1]))),
                                      (int(float(bbox[2])),
                                       int(float(bbox[3]))), (1.0, 0, 0), 2)
    return image


# tf.random.set_seed(22)
# np.random.seed(22)
# train_dataset = myDataset.myDataSet(flip_ratio=0.5,scale=(768, 768))
# num_classes = len(train_dataset.get_categories())
#
# img, img_meta, bboxes, labels = train_dataset[6]
# batch_imgs = tf.convert_to_tensor(np.expand_dims(img.astype(np.float32), 0))
# batch_metas = tf.convert_to_tensor(np.expand_dims(img_meta.astype(np.float32), 0))
# batch_bboxes = tf.convert_to_tensor(np.expand_dims(bboxes.astype(np.float32), 0))
# batch_labels = tf.convert_to_tensor(np.expand_dims(labels.astype(np.int), 0))
#
# image = showLabRpn(batch_imgs, batch_metas, batch_bboxes, batch_labels)
# plt.imshow(image)
# %%
