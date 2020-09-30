# %%
from functools import wraps

import numpy as np
import tensorflow as tf
from functools import reduce
from PIL import Image
from PIL import Image, ImageFont, ImageDraw


def letterbox_image(image, size):
    iw, ih = image.size
    w, h = size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128, 128, 128))
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
    return new_image


def compose(*funcs):
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')


# --------------------------------------------------#
#   单次卷积
# --------------------------------------------------#
@wraps(tf.keras.layers.Conv2D)
def DarknetConv2D(*args, **kwargs):
    darknet_conv_kwargs = {'kernel_regularizer': tf.keras.regularizers.l2(5e-4)}
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides') == (2, 2) else 'same'
    darknet_conv_kwargs.update(kwargs)
    return tf.keras.layers.Conv2D(*args, **darknet_conv_kwargs)


# ---------------------------------------------------#
#   卷积块
#   DarknetConv2D + BatchNormalization + LeakyReLU
# ---------------------------------------------------#
def DarknetConv2D_BN_Leaky(*args, **kwargs):
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(alpha=0.1))


def route_group(input_layer, groups, group_id):
    # 对通道数进行均等分割，我们取第二部分
    convs = tf.split(input_layer, num_or_size_splits=groups, axis=-1)
    return convs[group_id]


# ---------------------------------------------------#
#   CSPdarknet的结构块
#   存在一个大残差边
#   这个大残差边绕过了很多的残差结构
# ---------------------------------------------------#
def resblock_body(x, num_filters):
    # 特征整合
    x = DarknetConv2D_BN_Leaky(num_filters, (3, 3))(x)
    # 残差边route
    route = x
    # 通道分割
    x = tf.keras.layers.Lambda(route_group, arguments={'groups': 2, 'group_id': 1})(x)
    x = DarknetConv2D_BN_Leaky(int(num_filters / 2), (3, 3))(x)

    # 小残差边route1
    route_1 = x
    x = DarknetConv2D_BN_Leaky(int(num_filters / 2), (3, 3))(x)
    # 堆叠
    x = tf.keras.layers.Concatenate()([x, route_1])

    x = DarknetConv2D_BN_Leaky(num_filters, (1, 1))(x)
    # 第三个resblockbody会引出来一个有效特征层分支
    feat = x
    # 连接
    x = tf.keras.layers.Concatenate()([route, x])
    x = tf.keras.layers.MaxPooling2D(pool_size=[2, 2], )(x)

    # 最后对通道数进行整合
    return x, feat


# ---------------------------------------------------#
#   darknet53 的主体部分
# ---------------------------------------------------#
def darknet_body(x):
    # 进行长和宽的压缩
    x = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))(x)
    # 416,416,3 -> 208,208,32
    x = DarknetConv2D_BN_Leaky(32, (3, 3), strides=(2, 2))(x)

    # 进行长和宽的压缩
    x = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))(x)
    # 208,208,32 -> 104,104,64
    x = DarknetConv2D_BN_Leaky(64, (3, 3), strides=(2, 2))(x)
    # 104,104,64 -> 52,52,128
    x, _ = resblock_body(x, num_filters=64)
    # 52,52,128 -> 26,26,256
    x, _ = resblock_body(x, num_filters=128)
    # 26,26,256 -> 13,13,512
    # feat1的shape = 26,26,256
    x, feat1 = resblock_body(x, num_filters=256)

    x = DarknetConv2D_BN_Leaky(512, (3, 3))(x)

    feat2 = x
    return feat1, feat2


# ---------------------------------------------------#
#   特征层->最后的输出
# ---------------------------------------------------#
def yolo_body(inputs, num_anchors, num_classes):
    # 生成darknet53的主干模型
    # 首先我们会获取到两个有效特征层
    # feat1 26x26x256
    # feat2 13x13x512
    feat1, feat2 = darknet_body(inputs)

    # 13x13x512 -> 13x13x256
    P5 = DarknetConv2D_BN_Leaky(256, (1, 1))(feat2)

    P5_output = DarknetConv2D_BN_Leaky(512, (3, 3))(P5)
    P5_output = DarknetConv2D(num_anchors * (num_classes + 5), (1, 1))(P5_output)

    # Conv+UpSampling2D 13x13x256 -> 26x26x128
    P5_upsample = compose(DarknetConv2D_BN_Leaky(128, (1, 1)), tf.keras.layers.UpSampling2D(2))(P5)

    # 26x26x(128+256) 26x26x384
    P4 = tf.keras.layers.Concatenate()([feat1, P5_upsample])

    P4_output = DarknetConv2D_BN_Leaky(256, (3, 3))(P4)
    P4_output = DarknetConv2D(num_anchors * (num_classes + 5), (1, 1))(P4_output)

    return [P5_output, P4_output]


# ---------------------------------------------------#
#   将预测值的每个特征层调成真实值
# ---------------------------------------------------#
def yolo_head(feats, anchors, num_classes, input_shape, calc_loss=False):
    num_anchors = len(anchors)
    # [1, 1, 1, num_anchors, 2]
    anchors_tensor = tf.keras.backend.reshape(tf.keras.backend.constant(anchors), [1, 1, 1, num_anchors, 2])

    # 获得x，y的网格
    # (13,13, 1, 2)
    grid_shape = tf.keras.backend.shape(feats)[1:3]  # height, width
    grid_y = tf.keras.backend.tile(
        tf.keras.backend.reshape(tf.keras.backend.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]),
        [1, grid_shape[1], 1, 1])
    grid_x = tf.keras.backend.tile(
        tf.keras.backend.reshape(tf.keras.backend.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]),
        [grid_shape[0], 1, 1, 1])
    grid = tf.keras.backend.concatenate([grid_x, grid_y])
    grid = tf.keras.backend.cast(grid, tf.keras.backend.dtype(feats))

    # (batch_size,13,13,3,85)
    feats = tf.keras.backend.reshape(feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])

    # 将预测值调成真实值
    # box_xy对应框的中心点
    # box_wh对应框的宽和高
    box_xy = (tf.keras.backend.sigmoid(feats[..., :2]) + grid) / tf.keras.backend.cast(grid_shape[..., ::-1],
                                                                                       tf.keras.backend.dtype(feats))
    box_wh = tf.keras.backend.exp(feats[..., 2:4]) * anchors_tensor / tf.keras.backend.cast(input_shape[..., ::-1],
                                                                                            tf.keras.backend.dtype(
                                                                                                feats))
    box_confidence = tf.keras.backend.sigmoid(feats[..., 4:5])
    box_class_probs = tf.keras.backend.sigmoid(feats[..., 5:])

    # 在计算loss的时候返回如下参数
    if calc_loss == True:
        return grid, feats, box_xy, box_wh
    return box_xy, box_wh, box_confidence, box_class_probs


# ---------------------------------------------------#
#   对box进行调整，使其符合真实图片的样子
# ---------------------------------------------------#
def yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape):
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]

    input_shape = tf.keras.backend.cast(input_shape, tf.keras.backend.dtype(box_yx))
    image_shape = tf.keras.backend.cast(image_shape, tf.keras.backend.dtype(box_yx))

    new_shape = tf.keras.backend.round(image_shape * tf.keras.backend.min(input_shape / image_shape))
    offset = (input_shape - new_shape) / 2. / input_shape
    scale = input_shape / new_shape

    box_yx = (box_yx - offset) * scale
    box_hw *= scale

    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes = tf.keras.backend.concatenate([
        box_mins[..., 0:1],  # y_min
        box_mins[..., 1:2],  # x_min
        box_maxes[..., 0:1],  # y_max
        box_maxes[..., 1:2]  # x_max
    ])

    boxes *= tf.keras.backend.concatenate([image_shape, image_shape])
    return boxes


# ---------------------------------------------------#
#   获取每个box和它的得分
# ---------------------------------------------------#
def yolo_boxes_and_scores(feats, anchors, num_classes, input_shape, image_shape):
    # 将预测值调成真实值
    # box_xy对应框的中心点
    # box_wh对应框的宽和高
    # -1,13,13,3,2; -1,13,13,3,2; -1,13,13,3,1; -1,13,13,3,80
    box_xy, box_wh, box_confidence, box_class_probs = yolo_head(feats, anchors, num_classes, input_shape)
    # 将box_xy、和box_wh调节成y_min,y_max,xmin,xmax
    boxes = yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape)
    # 获得得分和box
    boxes = tf.keras.backend.reshape(boxes, [-1, 4])
    box_scores = box_confidence * box_class_probs
    box_scores = tf.keras.backend.reshape(box_scores, [-1, num_classes])
    return boxes, box_scores


# ---------------------------------------------------#
#   图片预测
# ---------------------------------------------------#
def yolo_eval(yolo_outputs,
              anchors,
              num_classes,
              image_shape,
              max_boxes=20,
              score_threshold=.7,
              iou_threshold=.5,
              eager=False):
    # image_shape = tf.keras.backend.reshape(yolo_outputs[-1],[-1])
    # num_layers = len(yolo_outputs)-1
    # num_layers = len(yolo_outputs)

    # 特征层1对应的anchor是678
    # 特征层2对应的anchor是345
    # 特征层3对应的anchor是012
    anchor_mask = [[3, 4, 5], [1, 2, 3]]

    input_shape = tf.keras.backend.shape(yolo_outputs[0])[1:3] * 32
    boxes = []
    box_scores = []
    # 对每个特征层进行处理
    for l in range(2):
        tf.print(l)
        _boxes, _box_scores = yolo_boxes_and_scores(yolo_outputs[l], anchors[anchor_mask[l]], num_classes, input_shape,
                                                    image_shape)
        boxes.append(_boxes)
        box_scores.append(_box_scores)
    # 将每个特征层的结果进行堆叠
    boxes = tf.keras.backend.concatenate(boxes, axis=0)
    box_scores = tf.keras.backend.concatenate(box_scores, axis=0)

    mask = box_scores >= score_threshold
    max_boxes_tensor = tf.keras.backend.constant(max_boxes, dtype='int32')
    boxes_ = []
    scores_ = []
    classes_ = []
    for c in range(num_classes):
        # 取出所有box_scores >= score_threshold的框，和成绩
        class_boxes = tf.boolean_mask(boxes, mask[:, c])
        class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])

        # 非极大抑制，去掉box重合程度高的那一些
        nms_index = tf.image.non_max_suppression(
            class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=iou_threshold)

        # 获取非极大抑制后的结果
        # 下列三个分别是
        # 框的位置，得分与种类
        class_boxes = tf.keras.backend.gather(class_boxes, nms_index)
        class_box_scores = tf.keras.backend.gather(class_box_scores, nms_index)
        classes = tf.keras.backend.ones_like(class_box_scores, 'int32') * c
        boxes_.append(class_boxes)
        scores_.append(class_box_scores)
        classes_.append(classes)
    boxes_ = tf.keras.backend.concatenate(boxes_, axis=0)
    scores_ = tf.keras.backend.concatenate(scores_, axis=0)
    classes_ = tf.keras.backend.concatenate(classes_, axis=0)

    return boxes_, scores_, classes_


imgInputs = tf.keras.layers.Input(shape=(416, 416, 3))
out = yolo_body(imgInputs, 3, 10)
model = tf.keras.models.Model(imgInputs, out)

# modelPath = r'logs\Epoch5-Total_Loss27.9598-Val_Loss23.5428.h5'
modelPath = r'logs\Epoch100-Total_Loss6.6205-Val_Loss5.7328.h5'

model.load_weights(modelPath)

# model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)

# tf.keras.utils.plot_model(model, show_shapes=True, show_layer_names=True)
# model.summary()
# freeze_layers = 60
# for i in range(freeze_layers): model.layers[i].trainable = False
# print('Freeze the first {} layers of total {} layers.'.format(freeze_layers, len(model.layers)))
# %%

# input_image_shape = tf.keras.layers.Input([2,],batch_size=1)
input_image_shape = tf.keras.layers.Input([2, ])
inputs = [*model.output, input_image_shape]

import os

anchors_path = 'yolo_anchors.txt'
with open(anchors_path) as f:
    anchors = f.readline()
anchors = [float(x) for x in anchors.split(',')]
anchors = np.array(anchors).reshape(-1, 2)
# , output_shape=(1,)

#
if True:
    image = Image.open('000001.jpg')



    boxed_image = letterbox_image(image, (416, 416))
    image_data = np.array(boxed_image, dtype='float32')
    image_data /= 255.
    image_data = np.expand_dims(image_data, 0)
    print(image_data.shape)
    input_image_shape = np.expand_dims(np.array([image.size[1], image.size[0]], dtype='float32'), 0)
    o = model.predict(image_data)
    # print(o)
    print(o[0].shape, o[1].shape)

    o1 = tf.constant(o[0])
    o2 = tf.constant(o[1])
    out_boxes, out_scores, out_classes = yolo_eval([o1, o2], anchors, 10, image_shape=tf.constant((416, 416)),
                                                   eager=True)
    print(out_boxes)

    # 画框设置不同的颜色
    hsv_tuples = [(x / 10, 1., 1.) for x in range(10)]
    import colorsys

    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(
        map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
            colors))

    print('Found {} boxes for {}'.format(len(out_boxes), 'img'))
    for i, c in list(enumerate(out_classes)):
        box = out_boxes[i]
        score = out_scores[i]

        top, left, bottom, right = box
        top = top - 5
        left = left - 5
        bottom = bottom + 5
        right = right + 5
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
        right = min(image.size[0], np.floor(right + 0.5).astype('int32'))

        draw = ImageDraw.Draw(image)
        tem = image.size[0] / 416.0
        draw.rectangle(
            [left * tem, top * tem, right * tem, bottom * tem],
            outline=colors[c])
        # thickness = (image.size[0] + image.size[1]) // 300
        # for i in range(thickness):
        #     draw.rectangle(
        #         [left + i, top + i, right - i, bottom - i],
        #         outline=colors[c])
        del draw
    image.show()
    print(image.size)



else:

    outputs = tf.keras.layers.Lambda(yolo_eval, name='yolo_eval',
                                     arguments={'anchors': anchors, 'num_classes': 20, 'image_shape': (416, 416),
                                                'score_threshold': 0.5, 'eager': True})(inputs)

    yolo_model = tf.keras.models.Model([model.input, input_image_shape], outputs)

    image = Image.open('street.jpg')
    boxed_image = letterbox_image(image, (416, 416))
    image_data = np.array(boxed_image, dtype='float32')
    image_data /= 255.
    image_data = np.expand_dims(image_data, 0)
    print(image_data.shape)
    # input_image_shape = np.expand_dims(np.array([image.size[1], image.size[0]], dtype='float32'), 0)
    input_image_shape = tf.constant([[416, 416]])
    out_boxes, out_scores, out_classes = yolo_model.predict([image_data, input_image_shape])
    # %%
    print(out_boxes)


