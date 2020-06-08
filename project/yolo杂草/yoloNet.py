# %%
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from tensorflow import keras
import tensorflow.keras.backend as K
from myconfig import cfg

class SpaceToDepth(layers.Layer):

    def __init__(self, block_size, **kwargs):
        self.block_size = block_size
        super(SpaceToDepth, self).__init__(**kwargs)

    def call(self, inputs):
        x = inputs
        batch, height, width, depth = K.int_shape(x)
        batch = -1
        reduced_height = height // self.block_size
        reduced_width = width // self.block_size
        y = K.reshape(x, (batch, reduced_height, self.block_size,
                          reduced_width, self.block_size, depth))
        z = K.permute_dimensions(y, (0, 1, 3, 2, 4, 5))
        t = K.reshape(z, (batch, reduced_height, reduced_width, depth * self.block_size ** 2))
        return t

    def compute_output_shape(self, input_shape):
        shape = (input_shape[0], input_shape[1] // self.block_size, input_shape[2] // self.block_size,
                 input_shape[3] * self.block_size ** 2)
        return tf.TensorShape(shape)



def yoloNetModle():
    IMGSZ = cfg.TRAIN.IMGSZ
    GRIDSZ = cfg.TRAIN.GRIDSZ
    scale = IMGSZ // GRIDSZ

    # 3.1
    input_image = layers.Input((IMGSZ, IMGSZ, 3), dtype='float32')

    # unit1
    x = layers.Conv2D(32, (3, 3), strides=(1, 1), padding='same', name='conv_1', use_bias=False)(input_image)
    x = layers.BatchNormalization(name='norm_1')(x)
    x = layers.LeakyReLU(alpha=0.1)(x)

    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    # unit2
    x = layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same', name='conv_2', use_bias=False)(x)
    x = layers.BatchNormalization(name='norm_2')(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    # Layer 3
    x = layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same', name='conv_3', use_bias=False)(x)
    x = layers.BatchNormalization(name='norm_3')(x)
    x = layers.LeakyReLU(alpha=0.1)(x)

    # Layer 4
    x = layers.Conv2D(64, (1, 1), strides=(1, 1), padding='same', name='conv_4', use_bias=False)(x)
    x = layers.BatchNormalization(name='norm_4')(x)
    x = layers.LeakyReLU(alpha=0.1)(x)

    # Layer 5
    x = layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same', name='conv_5', use_bias=False)(x)
    x = layers.BatchNormalization(name='norm_5')(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    # Layer 6
    x = layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same', name='conv_6', use_bias=False)(x)
    x = layers.BatchNormalization(name='norm_6')(x)
    x = layers.LeakyReLU(alpha=0.1)(x)

    # Layer 7
    x = layers.Conv2D(128, (1, 1), strides=(1, 1), padding='same', name='conv_7', use_bias=False)(x)
    x = layers.BatchNormalization(name='norm_7')(x)
    x = layers.LeakyReLU(alpha=0.1)(x)

    # Layer 8
    x = layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same', name='conv_8', use_bias=False)(x)
    x = layers.BatchNormalization(name='norm_8')(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    # Layer 9
    x = layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same', name='conv_9', use_bias=False)(x)
    x = layers.BatchNormalization(name='norm_9')(x)
    x = layers.LeakyReLU(alpha=0.1)(x)

    # Layer 10
    x = layers.Conv2D(256, (1, 1), strides=(1, 1), padding='same', name='conv_10', use_bias=False)(x)
    x = layers.BatchNormalization(name='norm_10')(x)
    x = layers.LeakyReLU(alpha=0.1)(x)

    # Layer 11
    x = layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same', name='conv_11', use_bias=False)(x)
    x = layers.BatchNormalization(name='norm_11')(x)
    x = layers.LeakyReLU(alpha=0.1)(x)

    # Layer 12
    x = layers.Conv2D(256, (1, 1), strides=(1, 1), padding='same', name='conv_12', use_bias=False)(x)
    x = layers.BatchNormalization(name='norm_12')(x)
    x = layers.LeakyReLU(alpha=0.1)(x)

    # Layer 13
    x = layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same', name='conv_13', use_bias=False)(x)
    x = layers.BatchNormalization(name='norm_13')(x)
    x = layers.LeakyReLU(alpha=0.1)(x)

    # for skip connection
    skip_x = x  # [b,32,32,512]

    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    # Layer 14
    x = layers.Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_14', use_bias=False)(x)
    x = layers.BatchNormalization(name='norm_14')(x)
    x = layers.LeakyReLU(alpha=0.1)(x)

    # Layer 15
    x = layers.Conv2D(512, (1, 1), strides=(1, 1), padding='same', name='conv_15', use_bias=False)(x)
    x = layers.BatchNormalization(name='norm_15')(x)
    x = layers.LeakyReLU(alpha=0.1)(x)

    # Layer 16
    x = layers.Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_16', use_bias=False)(x)
    x = layers.BatchNormalization(name='norm_16')(x)
    x = layers.LeakyReLU(alpha=0.1)(x)

    # Layer 17
    x = layers.Conv2D(512, (1, 1), strides=(1, 1), padding='same', name='conv_17', use_bias=False)(x)
    x = layers.BatchNormalization(name='norm_17')(x)
    x = layers.LeakyReLU(alpha=0.1)(x)

    # Layer 18
    x = layers.Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_18', use_bias=False)(x)
    x = layers.BatchNormalization(name='norm_18')(x)
    x = layers.LeakyReLU(alpha=0.1)(x)

    # Layer 19
    x = layers.Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_19', use_bias=False)(x)
    x = layers.BatchNormalization(name='norm_19')(x)
    x = layers.LeakyReLU(alpha=0.1)(x)

    # Layer 20
    x = layers.Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_20', use_bias=False)(x)
    x = layers.BatchNormalization(name='norm_20')(x)
    x = layers.LeakyReLU(alpha=0.1)(x)

    # Layer 21
    skip_x = layers.Conv2D(64, (1, 1), strides=(1, 1), padding='same', name='conv_21', use_bias=False)(skip_x)
    skip_x = layers.BatchNormalization(name='norm_21')(skip_x)
    skip_x = layers.LeakyReLU(alpha=0.1)(skip_x)

    skip_x = SpaceToDepth(block_size=2)(skip_x)

    # concat
    # [b,16,16,1024], [b,16,16,256],=> [b,16,16,1280]
    x = tf.concat([skip_x, x], axis=-1)

    # Layer 22
    x = layers.Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_22', use_bias=False)(x)
    x = layers.BatchNormalization(name='norm_22')(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.Dropout(0.5)(x)  # add dropout
    # [b,16,16,5,7] => [b,16,16,35]

    x = layers.Conv2D(5 * 7, (1, 1), strides=(1, 1), padding='same', name='conv_23')(x)

    output = layers.Reshape((GRIDSZ, GRIDSZ, 5, 7))(x)
    # create model
    model = keras.models.Model(input_image, output)
    # x = tf.random.normal((4, 512, 512, 3))
    # out = model(x)
    # print('out:', out.shape)
    return model


class WeightReader:
    def __init__(self, weight_file):
        self.offset = 4
        self.all_weights = np.fromfile(weight_file, dtype='float32')

    def read_bytes(self, size):
        self.offset = self.offset + size
        return self.all_weights[self.offset - size:self.offset]

    def reset(self):
        self.offset = 4


def loadWeight(model,path):
    IMGSZ = cfg.TRAIN.IMGSZ
    GRIDSZ = cfg.TRAIN.GRIDSZ

    weight_reader = WeightReader(path)
    weight_reader.reset()
    nb_conv = 23

    for i in range(1, nb_conv + 1):
        conv_layer = model.get_layer('conv_' + str(i))
        conv_layer.trainable = True

        if i < nb_conv:
            norm_layer = model.get_layer('norm_' + str(i))
            norm_layer.trainable = True

            size = np.prod(norm_layer.get_weights()[0].shape)

            beta = weight_reader.read_bytes(size)
            gamma = weight_reader.read_bytes(size)
            mean = weight_reader.read_bytes(size)
            var = weight_reader.read_bytes(size)

            weights = norm_layer.set_weights([gamma, beta, mean, var])

        if len(conv_layer.get_weights()) > 1:
            bias = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[1].shape))
            kernel = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
            kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
            kernel = kernel.transpose([2, 3, 1, 0])
            conv_layer.set_weights([kernel, bias])
        else:
            kernel = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
            kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
            kernel = kernel.transpose([2, 3, 1, 0])
            conv_layer.set_weights([kernel])

    layer = model.layers[-2]  # last convolutional layer
    # print(layer.name)
    layer.trainable = True

    weights = layer.get_weights()

    new_kernel = np.random.normal(size=weights[0].shape) / (GRIDSZ * GRIDSZ)
    new_bias = np.random.normal(size=weights[1].shape) / (GRIDSZ * GRIDSZ)

    layer.set_weights([new_kernel, new_bias])

    return model

def compute_iou(x1,y1,w1,h1, x2,y2,w2,h2):
    # x1...:[b,16,16,5]
    xmin1 = x1 - 0.5*w1
    xmax1 = x1 + 0.5*w1
    ymin1 = y1 - 0.5*h1
    ymax1 = y1 + 0.5*h1

    xmin2 = x2 - 0.5*w2
    xmax2 = x2 + 0.5*w2
    ymin2 = y2 - 0.5*h2
    ymax2 = y2 + 0.5*h2

    # (xmin1,ymin1,xmax1,ymax1) (xmin2,ymin2,xmax2,ymax2)
    interw = np.minimum(xmax1,xmax2) - np.maximum(xmin1,xmin2)
    interh = np.minimum(ymax1, ymax2) - np.maximum(ymin1, ymin2)
    inter = interw * interh
    union = w1*h1 +w2*h2 - inter
    iou = inter / (union + 1e-6)
    # [b,16,16,5]
    return iou

from tensorflow.keras import losses
# 4.1 coordinate loss
def yolo_loss(detector_mask, matching_gt_boxes, matching_classes_oh, gt_boxes_grid, y_pred):
    # detector_mask: [b,16,16,5,1]
    # matching_gt_boxes: [b,16,16,5,5] x-y-w-h-l
    # matching_classes_oh: [b,16,16,5,2] l1-l2
    # gt_boxes_grid: [b,40,5] x-y-wh-l
    # y_pred: [b,16,16,5,7] x-y-w-h-conf-l0-l1

    IMGSZ = cfg.TRAIN.IMGSZ
    GRIDSZ = cfg.TRAIN.GRIDSZ
    ANCHORS = cfg.TRAIN.ANCHORS

    anchors = np.array(ANCHORS).reshape(5, 2)

    # create starting position for each grid anchors
    # [16,16]
    x_grid = tf.tile(tf.range(GRIDSZ), [GRIDSZ])
    # [1,16,16,1,1]
    # [b,16,16,5,2]
    x_grid = tf.reshape(x_grid, (1, GRIDSZ, GRIDSZ, 1, 1))
    x_grid = tf.cast(x_grid, tf.float32)
    # [b,16_1,16_2,1,1]=>[b,16_2,16_1,1,1]
    y_grid = tf.transpose(x_grid, (0, 2, 1, 3, 4))
    xy_grid = tf.concat([x_grid, y_grid], axis=-1)
    # [1,16,16,1,2]=> [b,16,16,5,2]
    xy_grid = tf.tile(xy_grid, [y_pred.shape[0], 1, 1, 5, 1])

    # [b,16,16,5,7] x-y-w-h-conf-l1-l2
    pred_xy = tf.sigmoid(y_pred[..., 0:2])
    pred_xy = pred_xy + xy_grid
    # [b,16,16,5,2]
    pred_wh = tf.exp(y_pred[..., 2:4])
    # [b,16,16,5,2] * [5,2] => [b,16,16,5,2]
    pred_wh = pred_wh * anchors

    n_detector_mask = tf.reduce_sum(tf.cast(detector_mask > 0., tf.float32))
    # [b,16,16,5,1] * [b,16,16,5,2]
    #
    xy_loss = detector_mask * tf.square(matching_gt_boxes[..., :2] - pred_xy) / (n_detector_mask + 1e-6)
    xy_loss = tf.reduce_sum(xy_loss)
    wh_loss = detector_mask * tf.square(tf.sqrt(matching_gt_boxes[..., 2:4]) - \
                                        tf.sqrt(pred_wh)) / (n_detector_mask + 1e-6)
    wh_loss = tf.reduce_sum(wh_loss)

    # 4.1 coordinate loss
    coord_loss = xy_loss + wh_loss

    # 4.2 class loss
    # [b,16,16,5,2]
    pred_box_class = y_pred[..., 5:]
    # [b,16,16,5]
    true_box_class = tf.argmax(matching_classes_oh, -1)
    # [b,16,16,5] vs [b,16,16,5,2]
    class_loss = losses.sparse_categorical_crossentropy( \
        true_box_class, pred_box_class, from_logits=True)
    # [b,16,16,5] => [b,16,16,5,1]* [b,16,16,5,1]
    class_loss = tf.expand_dims(class_loss, -1) * detector_mask
    class_loss = tf.reduce_sum(class_loss) / (n_detector_mask + 1e-6)

    # 4.3 object loss
    # nonobject_mask
    # iou done!
    # [b,16,16,5]
    x1, y1, w1, h1 = matching_gt_boxes[..., 0], matching_gt_boxes[..., 1], \
                     matching_gt_boxes[..., 2], matching_gt_boxes[..., 3]
    # [b,16,16,5]
    x2, y2, w2, h2 = pred_xy[..., 0], pred_xy[..., 1], pred_wh[..., 0], pred_wh[..., 1]
    ious = compute_iou(x1, y1, w1, h1, x2, y2, w2, h2)
    # [b,16,16,5,1]
    ious = tf.expand_dims(ious, axis=-1)

    # [b,16,16,5,1]
    pred_conf = tf.sigmoid(y_pred[..., 4:5])
    # [b,16,16,5,2] => [b,16,16,5, 1, 2]
    pred_xy = tf.expand_dims(pred_xy, axis=4)
    # [b,16,16,5,2] => [b,16,16,5, 1, 2]
    pred_wh = tf.expand_dims(pred_wh, axis=4)
    pred_wh_half = pred_wh / 2.
    pred_xymin = pred_xy - pred_wh_half
    pred_xymax = pred_xy + pred_wh_half

    # [b, 40, 5] => [b, 1, 1, 1, 40, 5]
    true_boxes_grid = tf.reshape(gt_boxes_grid, [gt_boxes_grid.shape[0], 1, 1, 1, gt_boxes_grid.shape[1], gt_boxes_grid.shape[2]])
    true_xy = true_boxes_grid[..., 0:2]
    true_wh = true_boxes_grid[..., 2:4]
    true_wh_half = true_wh / 2.
    true_xymin = true_xy - true_wh_half
    true_xymax = true_xy + true_wh_half
    # predxymin, predxymax, true_xymin, true_xymax
    # [b,16,16,5,1,2] vs [b,1,1,1,40,2]=> [b,16,16,5,40,2]
    intersectxymin = tf.maximum(pred_xymin, true_xymin)
    # [b,16,16,5,1,2] vs [b,1,1,1,40,2]=> [b,16,16,5,40,2]
    intersectxymax = tf.minimum(pred_xymax, true_xymax)
    # [b,16,16,5,40,2]
    intersect_wh = tf.maximum(intersectxymax - intersectxymin, 0.)
    # [b,16,16,5,40] * [b,16,16,5,40]=>[b,16,16,5,40]
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    # [b,16,16,5,1]
    pred_area = pred_wh[..., 0] * pred_wh[..., 1]
    # [b,1,1,1,40]
    true_area = true_wh[..., 0] * true_wh[..., 1]
    # [b,16,16,5,1]+[b,1,1,1,40]-[b,16,16,5,40]=>[b,16,16,5,40]
    union_area = pred_area + true_area - intersect_area
    # [b,16,16,5,40]
    iou_score = intersect_area / union_area
    # [b,16,16,5]
    best_iou = tf.reduce_max(iou_score, axis=4)
    # [b,16,16,5,1]
    best_iou = tf.expand_dims(best_iou, axis=-1)

    nonobj_detection = tf.cast(best_iou < 0.6, tf.float32)
    nonobj_mask = nonobj_detection * (1 - detector_mask)
    # nonobj counter
    n_nonobj = tf.reduce_sum(tf.cast(nonobj_mask > 0., tf.float32))

    nonobj_loss = tf.reduce_sum(nonobj_mask * tf.square(-pred_conf)) \
                  / (n_nonobj + 1e-6)
    obj_loss = tf.reduce_sum(detector_mask * tf.square(ious - pred_conf)) \
               / (n_detector_mask + 1e-6)

    loss = coord_loss + class_loss + nonobj_loss + 5 * obj_loss

    return loss, [nonobj_loss + 5 * obj_loss, class_loss, coord_loss]