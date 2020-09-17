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



tf.random.set_seed(22)
np.random.seed(22)

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


train_dataset = myDataset.myDataSet(flip_ratio=0.5,scale=(768, 768))
num_classes = len(train_dataset.get_categories())
train_generator = data_generator.DataGenerator(train_dataset)
train_tf_dataset = tf.data.Dataset.from_generator(train_generator, (tf.float32, tf.float32, tf.float32, tf.int32))
train_tf_dataset = train_tf_dataset.batch(2).prefetch(100).shuffle(100)
model = faster_rcnn2.FasterRCNN(num_classes=num_classes)

# optimizer = keras.optimizers.SGD(1e-3, momentum=0.9, nesterov=True)
optimizer = keras.optimizers.Adam(0.0001)
print([var.name for var in model.trainable_variables])

img, img_meta, bboxes, labels = train_dataset[6]
batch_imgs = tf.convert_to_tensor(np.expand_dims(img.astype(np.float32), 0))
batch_metas = tf.convert_to_tensor(np.expand_dims(img_meta.astype(np.float32), 0))
batch_bboxes = tf.convert_to_tensor(np.expand_dims(bboxes.astype(np.float32), 0))
batch_labels = tf.convert_to_tensor(np.expand_dims(labels.astype(np.int), 0))
#%%
rpn_class_loss, rpn_bbox_loss = model((batch_imgs, batch_metas, batch_bboxes, batch_labels), training=True)
# model.load_weights('weights/faster_rcnn.h5', by_name=True)
#%%

# rpn_class_logits, rpn_probs = model((batch_imgs, batch_metas),training=False)
#
# anchors, valid_flags = model.rpn_head.generator.generate_pyramid_anchors(batch_metas)
#
# rpn_probs = rpn_probs[0, :, 1]
# valid_flags = valid_flags[0]
# print(rpn_probs.shape)
# print(anchors.shape)
# print(valid_flags.shape)
# valid_flags = tf.cast(valid_flags, tf.bool)
# rpn_probs = tf.boolean_mask(rpn_probs, valid_flags)
# anchors = tf.boolean_mask(anchors, valid_flags)
# ix = tf.nn.top_k(rpn_probs, 100, sorted=True).indices
# rpn_probs = tf.gather(rpn_probs, ix)
# anchors = tf.gather(anchors, ix)
# print(rpn_probs)
# print(anchors)
#
# image = batch_imgs[0].numpy()
# print(image)
# bboxs = anchors.numpy()
# for i in range(bboxs.shape[0]):
#     bbox = bboxs[i]# *768
#     print(bbox)
#     image = cv2.rectangle(image, (int(float(bbox[0])),
#                                   int(float(bbox[1]))),
#                                   (int(float(bbox[2])),
#                                    int(float(bbox[3]))), (1.0, 0, 0), 2)
# cv2.imshow('img', image)
# cv2.waitKey(0)
#
# exit()
#%%
# for (batch, inputs) in enumerate(train_tf_dataset):
#     batch_imgs, batch_metas, batch_bboxes, batch_labels = inputs
#     with tf.GradientTape() as tape:
#         rpn_class_loss, rpn_bbox_loss = model((batch_imgs, batch_metas, batch_bboxes, batch_labels), training=True)
#     break
# print([var.name for var in model.trainable_variables])



for epoch in range(100):
    loss_history = []
    for (batch, inputs) in enumerate(train_tf_dataset):
        batch_imgs, batch_metas, batch_bboxes, batch_labels = inputs
        with tf.GradientTape() as tape:
            rpn_class_loss, rpn_bbox_loss = model((batch_imgs, batch_metas, batch_bboxes, batch_labels), training=True) # , rcnn_class_loss, rcnn_bbox_loss

            loss_value = rpn_class_loss  + rpn_bbox_loss*0.1 # + rcnn_class_loss + rcnn_bbox_loss

        grads = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        loss_history.append(loss_value.numpy())

        if batch % 100 == 0:
            print(rpn_class_loss, rpn_bbox_loss) # , rcnn_class_loss, rcnn_bbox_loss
            print('epoch', epoch, batch, np.mean(loss_history))
            model.save_weights('weights/faster_rcnn0_0.h5')

