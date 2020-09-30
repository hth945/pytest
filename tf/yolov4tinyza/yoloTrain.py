# %%
from functools import wraps

import numpy as np
import tensorflow as tf
from functools import reduce
from PIL import Image
from myYoloData import data_generator, get_anchors,get_classes
from functools import partial
import time
from tqdm import tqdm
from loss import yolo_loss

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
              image_shape=(1,416,416),
              max_boxes=20,
              score_threshold=.6,
              iou_threshold=.5):
    image_shape = tf.constant(image_shape)
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


# @tf.function
def train_step(imgs, yolo_loss, targets, net, optimizer, regularization):
    with tf.GradientTape() as tape:
        # 计算loss
        P5_output, P4_output = net(imgs, training=True)
        args = [P5_output, P4_output] + targets
        loss_value = yolo_loss(args, anchors, num_classes, label_smoothing=label_smoothing)
        if regularization:
            # 加入正则化损失
            loss_value = tf.reduce_sum(net.losses) + loss_value
    grads = tape.gradient(loss_value, net.trainable_variables)
    optimizer.apply_gradients(zip(grads, net.trainable_variables))
    return loss_value


def fit_one_epoch(net, yolo_loss, optimizer, epoch, epoch_size, epoch_size_val, gen, genval, Epoch, anchors,
                  num_classes, label_smoothing, regularization=False):
    loss = 0
    val_loss = 0
    start_time = time.time()
    with tqdm(total=epoch_size, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_size:
                break
            images, target0, target1 = batch[0], batch[1], batch[2]
            targets = [target0, target1]
            targets = [tf.convert_to_tensor(target) for target in targets]
            loss_value = train_step(images, yolo_loss, targets, net, optimizer, regularization)
            loss = loss + loss_value

            waste_time = time.time() - start_time
            pbar.set_postfix(**{'total_loss': float(loss) / (iteration + 1),
                                'step/s': waste_time})
            pbar.update(1)
            start_time = time.time()

    print('Start Validation')
    with tqdm(total=epoch_size_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(genval):
            if iteration >= epoch_size_val:
                break
            # 计算验证集loss
            images, target0, target1 = batch[0], batch[1], batch[2]
            targets = [target0, target1]
            targets = [tf.convert_to_tensor(target) for target in targets]

            P5_output, P4_output = net(images)
            args = [P5_output, P4_output] + targets
            loss_value = yolo_loss(args, anchors, num_classes, label_smoothing=label_smoothing)
            if regularization:
                # 加入正则化损失
                loss_value = tf.reduce_sum(net.losses) + loss_value
            # 更新验证集loss
            val_loss = val_loss + loss_value

            pbar.set_postfix(**{'total_loss': float(val_loss) / (iteration + 1)})
            pbar.update(1)

    print('Finish Validation')
    print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
    print('Total Loss: %.4f || Val Loss: %.4f ' % (loss / (epoch_size + 1), val_loss / (epoch_size_val + 1)))
    net.save_weights('logs/Epoch%d-Total_Loss%.4f-Val_Loss%.4f.h5' % (
    (epoch + 1), loss / (epoch_size + 1), val_loss / (epoch_size_val + 1)))



gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)



annotation_path =  r'D:\sysDef\Documents\GitHub\pytest\dataAndModel\data\mnist\objtrainlab.txt'
classes_path = 'classes.txt'
anchors_path = 'yolo_anchors.txt'
# weights_path = r'D:\sysDef\Documents\GitHub\yolov4-tiny-tf2\model_data\yolov4_tiny_weights_coco.h5'
weights_path = r'logs\Epoch100-Total_Loss11.3523-Val_Loss10.2685.h5'
class_names = get_classes(classes_path)
anchors = get_anchors(anchors_path)
num_classes = len(class_names)
num_anchors = len(anchors)
input_shape = (416,416)

mosaic = False
Cosine_scheduler = False
label_smoothing = 0
# 是否使用正则化
regularization = True
# -------------------------------#
#   Dataloder的使用
# -------------------------------#
Use_Data_Loader = True

# 输入的图像为
image_input = tf.keras.layers.Input(shape=(None, None, 3))
h, w = input_shape

# 创建yolo模型
print('Create YOLOv4 model with {} anchors and {} classes.'.format(num_anchors, num_classes))
out = yolo_body(image_input, num_anchors // 2, num_classes)
model_body = tf.keras.models.Model(image_input, out)
print('Load weights {}.'.format(weights_path))
model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)

# 0.1用于验证，0.9用于训练
val_split = 0.1
with open(annotation_path) as f:
    lines = f.readlines()
np.random.seed(10101)
np.random.shuffle(lines)
np.random.seed(None)
num_val = int(len(lines) * val_split)
num_train = len(lines) - num_val

freeze_layers = 60
for i in range(freeze_layers): model_body.layers[i].trainable = False
print('Freeze the first {} layers of total {} layers.'.format(freeze_layers, len(model_body.layers)))



if False:
    Init_epoch = 0
    Freeze_epoch = 5
    # batch_size大小，每次喂入多少数据
    batch_size = 16
    # 最大学习率
    learning_rate_base = 1e-3
    if Use_Data_Loader:
        gen = partial(data_generator, annotation_lines=lines[:num_train], batch_size=batch_size,
                      input_shape=input_shape,
                      anchors=anchors, num_classes=num_classes, mosaic=mosaic)
        gen = tf.data.Dataset.from_generator(gen, (tf.float32, tf.float32, tf.float32))

        gen_val = partial(data_generator, annotation_lines=lines[num_train:], batch_size=batch_size,
                          input_shape=input_shape, anchors=anchors, num_classes=num_classes, mosaic=False)
        gen_val = tf.data.Dataset.from_generator(gen_val, (tf.float32, tf.float32, tf.float32))

        gen = gen.shuffle(buffer_size=batch_size).prefetch(buffer_size=batch_size)
        gen_val = gen_val.shuffle(buffer_size=batch_size).prefetch(buffer_size=batch_size)

    else:
        gen = data_generator(lines[:num_train], batch_size, input_shape, anchors, num_classes, mosaic=mosaic)
        gen_val = data_generator(lines[num_train:], batch_size, input_shape, anchors, num_classes, mosaic=False)

    epoch_size = num_train // batch_size
    epoch_size_val = num_val // batch_size

    if Cosine_scheduler:
        lr_schedule = tf.keras.experimental.CosineDecayRestarts(
            initial_learning_rate=learning_rate_base,
            first_decay_steps=5 * epoch_size,
            t_mul=1.0,
            alpha=1e-2
        )
    else:
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=learning_rate_base,
            decay_steps=epoch_size,
            decay_rate=0.9,
            staircase=True
        )
    print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    for epoch in range(Init_epoch, Freeze_epoch):
        fit_one_epoch(model_body, yolo_loss, optimizer, epoch, epoch_size, epoch_size_val, gen, gen_val,
                      Freeze_epoch, anchors, num_classes, label_smoothing, regularization)

for i in range(freeze_layers): model_body.layers[i].trainable = True

# 解冻后训练
if True:
    Freeze_epoch = 50
    Epoch = 100
    # batch_size大小，每次喂入多少数据
    batch_size = 16
    # 最大学习率
    learning_rate_base = 1e-4
    if Use_Data_Loader:
        gen = partial(data_generator, annotation_lines=lines[:num_train], batch_size=batch_size,
                      input_shape=input_shape,
                      anchors=anchors, num_classes=num_classes, mosaic=mosaic)
        gen = tf.data.Dataset.from_generator(gen, (tf.float32, tf.float32, tf.float32))

        gen_val = partial(data_generator, annotation_lines=lines[num_train:], batch_size=batch_size,
                          input_shape=input_shape, anchors=anchors, num_classes=num_classes, mosaic=False)
        gen_val = tf.data.Dataset.from_generator(gen_val, (tf.float32, tf.float32, tf.float32))

        gen = gen.shuffle(buffer_size=batch_size).prefetch(buffer_size=batch_size)
        gen_val = gen_val.shuffle(buffer_size=batch_size).prefetch(buffer_size=batch_size)
    else:
        gen = data_generator(lines[:num_train], batch_size, input_shape, anchors, num_classes, mosaic=mosaic)
        gen_val = data_generator(lines[num_train:], batch_size, input_shape, anchors, num_classes, mosaic=False)

    epoch_size = num_train // batch_size
    epoch_size_val = num_val // batch_size
    if Cosine_scheduler:
        lr_schedule = tf.keras.experimental.CosineDecayRestarts(
            initial_learning_rate=learning_rate_base,
            first_decay_steps=5 * epoch_size,
            t_mul=1.0,
            alpha=1e-2
        )
    else:
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=learning_rate_base,
            decay_steps=epoch_size,
            decay_rate=0.9,
            staircase=True
        )
    print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    for epoch in range(Freeze_epoch, Epoch):
        fit_one_epoch(model_body, yolo_loss, optimizer, epoch, epoch_size, epoch_size_val, gen, gen_val,
                      Epoch, anchors, num_classes, label_smoothing, regularization)