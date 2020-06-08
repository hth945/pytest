# %%
import os, glob

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow as tf
from tensorflow import keras
import xml.etree.ElementTree as ET

from matplotlib import pyplot as plt
from matplotlib import patches
from myconfig import cfg


tf.random.set_seed(2234)
np.random.seed(2234)

print(tf.__version__)
# print(tf.test.is_gpu_available())


def parse_annotation(img_dir, ann_dir, labels):
    # img_dir: image path
    # ann_dir: annotation xml file path
    # labels: ('sugarweet', 'weed')
    # parse annotation info from xml file
    """
    <annotation>
        <object>
            <name>sugarbeet</name>
            <bndbox>
                <xmin>1</xmin>
                <ymin>250</ymin>
                <xmax>53</xmax>
                <ymax>289</ymax>
            </bndbox>
        </object>
        <object>....

    """
    imgs_info = []

    max_boxes = 0
    # for each annotation xml file
    for ann in os.listdir(ann_dir):
        tree = ET.parse(os.path.join(ann_dir, ann))

        img_info = dict()
        boxes_counter = 0
        img_info['object'] = []
        for elem in tree.iter():
            if 'filename' in elem.tag:
                img_info['filename'] = os.path.join(img_dir, elem.text)
            if 'width' in elem.tag:
                img_info['width'] = int(elem.text)
                assert img_info['width'] == 512
            if 'height' in elem.tag:
                img_info['height'] = int(elem.text)
                assert img_info['height'] == 512

            if 'object' in elem.tag or 'part' in elem.tag:
                # x1-y1-x2-y2-label
                object_info = [0, 0, 0, 0, 0]
                boxes_counter += 1
                for attr in list(elem):
                    if 'name' in attr.tag:
                        label = labels.index(attr.text) + 1
                        object_info[4] = label
                    if 'bndbox' in attr.tag:
                        for pos in list(attr):
                            if 'xmin' in pos.tag:
                                object_info[0] = int(pos.text)
                            if 'ymin' in pos.tag:
                                object_info[1] = int(pos.text)
                            if 'xmax' in pos.tag:
                                object_info[2] = int(pos.text)
                            if 'ymax' in pos.tag:
                                object_info[3] = int(pos.text)
                img_info['object'].append(object_info)

        imgs_info.append(img_info)  # filename, w/h/box_info
        # (N,5)=(max_objects_num, 5)
        if boxes_counter > max_boxes:
            max_boxes = boxes_counter
    # the maximum boxes number is max_boxes
    # [b, 40, 5]
    boxes = np.zeros([len(imgs_info), max_boxes, 5])
    print(boxes.shape)
    imgs = []  # filename list
    for i, img_info in enumerate(imgs_info):
        # [N,5]
        img_boxes = np.array(img_info['object'])
        # overwrite the N boxes info
        boxes[i, :img_boxes.shape[0]] = img_boxes

        imgs.append(img_info['filename'])

        # print(img_info['filename'], boxes[i,:5])
    # imgs: list of image path
    # boxes: [b,40,5]
    return imgs, boxes


def preprocess(img, img_boxes):
    # img: string
    # img_boxes: [40,5]
    x = tf.io.read_file(img)
    x = tf.image.decode_png(x, channels=3)
    x = tf.image.convert_image_dtype(x, tf.float32)

    return x, img_boxes


# 1.2
def get_dataset(img_dir, ann_dir, batchsz, obj_names):
    # return tf dataset
    # [b], boxes [b, 40, 5]
    imgs, boxes = parse_annotation(img_dir, ann_dir, obj_names)
    db = tf.data.Dataset.from_tensor_slices((imgs, boxes))
    db = db.shuffle(1000).map(preprocess).batch(batchsz).repeat()

    print('db Images:', len(imgs))

    return db


def db_visualize(db):
    # imgs:[b, 512, 512, 3]
    # imgs_boxes: [b, 40, 5]
    imgs, imgs_boxes = next(iter(db))
    img, img_boxes = imgs[0], imgs_boxes[0]

    f, ax1 = plt.subplots(1, figsize=(10, 10))
    # display the image, [512,512,3]
    ax1.imshow(img)
    for x1, y1, x2, y2, l in img_boxes:  # [40,5]
        x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
        w = x2 - x1
        h = y2 - y1

        if l == 1:  # green for sugarweet
            color = (0, 1, 0)
        elif l == 2:  # red for weed
            color = (1, 0, 0)  # (R,G,B)
        else:  # ignore invalid boxes
            break

        rect = patches.Rectangle((x1, y1), w, h, linewidth=2,
                                 edgecolor=color, facecolor='none')
        ax1.add_patch(rect)
    plt.show()


# %%
# 1.4 data augmentation
import imgaug as ia
from imgaug import augmenters as iaa


def augmentation_generator(yolo_dataset):
    '''
    Augmented batch generator from a yolo dataset

    Parameters
    ----------
    - YOLO dataset

    Returns
    -------
    - augmented batch : tensor (shape : batch_size, IMAGE_W, IMAGE_H, 3)
        batch : tupple(images, annotations)
        batch[0] : images : tensor (shape : batch_size, IMAGE_W, IMAGE_H, 3)
        batch[1] : annotations : tensor (shape : batch_size, max annot, 5)
    '''
    for batch in yolo_dataset:
        # conversion tensor->numpy
        img = batch[0].numpy()
        boxes = batch[1].numpy()
        # conversion bbox numpy->ia object
        ia_boxes = []
        for i in range(img.shape[0]):
            ia_bbs = [ia.BoundingBox(x1=bb[0],
                                     y1=bb[1],
                                     x2=bb[2],
                                     y2=bb[3]) for bb in boxes[i]
                      if (bb[0] + bb[1] + bb[2] + bb[3] > 0)]
            ia_boxes.append(ia.BoundingBoxesOnImage(ia_bbs, shape=(512, 512)))
        # data augmentation
        seq = iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
            iaa.Multiply((0.4, 1.6)),  # change brightness
            # iaa.ContrastNormalization((0.5, 1.5)),
            # iaa.Affine(translate_px={"x": (-100,100), "y": (-100,100)}, scale=(0.7, 1.30))
        ])
        # seq = iaa.Sequential([])
        seq_det = seq.to_deterministic()
        img_aug = seq_det.augment_images(img)
        img_aug = np.clip(img_aug, 0, 1)
        boxes_aug = seq_det.augment_bounding_boxes(ia_boxes)
        # conversion ia object -> bbox numpy
        for i in range(img.shape[0]):
            boxes_aug[i] = boxes_aug[i].remove_out_of_image().clip_out_of_image()
            for j, bb in enumerate(boxes_aug[i].bounding_boxes):
                boxes[i, j, 0] = bb.x1
                boxes[i, j, 1] = bb.y1
                boxes[i, j, 2] = bb.x2
                boxes[i, j, 3] = bb.y2
        # conversion numpy->tensor
        batch = (tf.convert_to_tensor(img_aug), tf.convert_to_tensor(boxes))
        # batch = (img_aug, boxes)
        yield batch


# %%
def process_true_boxes(gt_boxes, anchors, ):
    # gt_boxes: [40,5]
    # 512//16=32
    IMGSZ = cfg.TRAIN.IMGSZ
    GRIDSZ = cfg.TRAIN.GRIDSZ
    scale = IMGSZ // GRIDSZ
    # [5,2]
    anchors = np.array(anchors).reshape((5, 2))

    # mask for object
    detector_mask = np.zeros([GRIDSZ, GRIDSZ, 5, 1])
    #x-y-w-h-l
    matching_gt_box = np.zeros([GRIDSZ, GRIDSZ, 5, 5])
    # [40,5] x1-y1-x2-y2-l => x-y-w-h-l
    gt_boxes_grid = np.zeros(gt_boxes.shape)
    # DB: tensor => numpy
    gt_boxes = gt_boxes.numpy()

    for i,box in enumerate(gt_boxes): # [40,5]
        # box: [5], x1-y1-x2-y2-l
        # 512 => 16
        x = ((box[0]+box[2])/2)/scale
        y = ((box[1]+box[3])/2)/scale
        w = (box[2] - box[0]) / scale
        h = (box[3] - box[1]) / scale
        # [40,5] x-y-w-h-l
        gt_boxes_grid[i] = np.array([x,y,w,h,box[4]])

        if w*h > 0: # valid box
            # x,y: 7.3, 6.8
            best_anchor = 0
            best_iou = 0
            for j in range(5):
                interct = np.minimum(w, anchors[j,0]) * np.minimum(h, anchors[j,1])
                union = w*h + (anchors[j,0]*anchors[j,1]) - interct
                iou = interct / union

                if iou > best_iou: # best iou
                    best_anchor = j
                    best_iou = iou
            # found the best anchors
            if best_iou>0:
               x_coord = np.floor(x).astype(np.int32)
               y_coord = np.floor(y).astype(np.int32)
               # [b,h,w,5,1]
               detector_mask[y_coord, x_coord, best_anchor] = 1
               # [b,h,w,5,x-y-w-h-l]
               matching_gt_box[y_coord, x_coord, best_anchor] = \
                   np.array([x,y,w,h,box[4]])

    # [40,5] => [16,16,5,5]
    # [16,16,5,5]
    # [16,16,5,1]
    # [40,5]
    return matching_gt_box, detector_mask, gt_boxes_grid

# %%
# 2.2
def ground_truth_generator(db):
    ANCHORS = cfg.TRAIN.ANCHORS
    for imgs, imgs_boxes in db:
        # imgs: [b,512,512,3]
        # imgs_boxes: [b,40,5]

        batch_matching_gt_box = []
        batch_detector_mask = []
        batch_gt_boxes_grid = []

        # print(imgs_boxes[0,:5])

        b = imgs.shape[0]
        for i in range(b): # for each image
            matching_gt_box, detector_mask, gt_boxes_grid = \
                process_true_boxes(imgs_boxes[i], ANCHORS)
            batch_matching_gt_box.append(matching_gt_box)
            batch_detector_mask.append(detector_mask)
            batch_gt_boxes_grid.append(gt_boxes_grid)
        # [b, 16,16,5,1]
        detector_mask = tf.cast(np.array(batch_detector_mask), dtype=tf.float32)
        # [b,16,16,5,5] x-y-w-h-l
        matching_gt_box = tf.cast(np.array(batch_matching_gt_box), dtype=tf.float32)
        # [b,40,5] x-y-w-h-l
        gt_boxes_grid = tf.cast(np.array(batch_gt_boxes_grid), dtype=tf.float32)

        # [b,16,16,5]
        matching_classes = tf.cast(matching_gt_box[...,4], dtype=tf.int32)
        # [b,16,16,5,3]
        matching_classes_oh = tf.one_hot(matching_classes, depth=3)
        # x-y-w-h-conf-l1-l2
        # [b,16,16,5,2]
        matching_classes_oh = tf.cast(matching_classes_oh[...,1:], dtype=tf.float32)


        # [b,512,512,3]
        # [b,16,16,5,1]
        # [b,16,16,5,5]
        # [b,16,16,5,2]
        # [b,40,5]
        yield imgs, detector_mask, matching_gt_box, matching_classes_oh,gt_boxes_grid