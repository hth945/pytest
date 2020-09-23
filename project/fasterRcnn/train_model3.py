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
from    detection.models import faster_rcnn3
import  tensorflow as tf
from    tensorflow import keras

tf.random.set_seed(22)
np.random.seed(22)

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


train_dataset = myDataset.myDataSet(flip_ratio=0.5,scale=(384, 384))
num_classes = len(train_dataset.get_categories())
train_generator = data_generator.DataGenerator(train_dataset)
train_tf_dataset = tf.data.Dataset.from_generator(train_generator, (tf.float32, tf.float32, tf.float32, tf.int32))
train_tf_dataset = train_tf_dataset.batch(2).prefetch(100).shuffle(100)
model = faster_rcnn3.FasterRCNN(num_classes=num_classes)

# optimizer = keras.optimizers.SGD(1e-3, momentum=0.9, nesterov=True)
optimizer = keras.optimizers.Adam(0.001)
print([var.name for var in model.trainable_variables])

# #####################all Test################################
# for i in range(20):
#     img, img_meta, bboxes, labels = train_dataset[i]
#     batch_imgs = tf.convert_to_tensor(np.expand_dims(img.astype(np.float32), 0))
#     batch_metas = tf.convert_to_tensor(np.expand_dims(img_meta.astype(np.float32), 0))
#     batch_bboxes = tf.convert_to_tensor(np.expand_dims(bboxes.astype(np.float32), 0))
#     batch_labels = tf.convert_to_tensor(np.expand_dims(labels.astype(np.int), 0))
#     #%%
#     if i == 0:
#         _ = model((batch_imgs, batch_metas, batch_bboxes, batch_labels), training=True)
#         model.load_weights('weights/faster_rcnn0_4.h5', by_name=True)
#         # _ = model((batch_imgs, batch_metas, batch_bboxes, batch_labels), training=True)
#         # tf.keras.utils.plot_model(model.rpn_head, show_shapes=True, show_layer_names=True)
#     #%%
#     ########################test#################################
#     # rois_list = model((batch_imgs, batch_metas),training=False)
#     rois_list,rois_list2 = model((batch_imgs, batch_metas, batch_bboxes, batch_labels), training=True,rec=2)
#
#     import imgTest
#
#     print(rois_list)
#     image = batch_imgs[0].numpy()
#     bboxs = rois_list[0].numpy()
#     for i in range(bboxs.shape[0]):
#         # if bboxs[i][4] < 0.9:
#         #     continue
#         bbox = bboxs[i]
#         image = cv2.rectangle(image, (int(float(bbox[0])),
#                                       int(float(bbox[1]))),
#                               (int(float(bbox[2])),
#                                int(float(bbox[3]))), (255, 0, 0), 2)
#     cv2.imshow('img', image)
#     img2 = imgTest.showLabRpn(batch_imgs, batch_metas, batch_bboxes, None)
#     cv2.imshow('img2', img2)
#
#     print(rois_list2)
#     image = batch_imgs[0].numpy()
#     bboxs = rois_list2[0].numpy()
#     for i in range(bboxs.shape[0]):
#         # if bboxs[i][4] < 0.9:
#         #     continue
#         bbox = bboxs[i]
#         image = cv2.rectangle(image, (int(float(bbox[0])),
#                                       int(float(bbox[1]))),
#                               (int(float(bbox[2])),
#                                int(float(bbox[3]))), (255, 0, 0), 2)
#     cv2.imshow('img3', image)
#
#
#     cv2.waitKey(0)

# #####################RPN Test################################
# for i in range(20):
#     img, img_meta, bboxes, labels = train_dataset[i]
#     batch_imgs = tf.convert_to_tensor(np.expand_dims(img.astype(np.float32), 0))
#     batch_metas = tf.convert_to_tensor(np.expand_dims(img_meta.astype(np.float32), 0))
#     batch_bboxes = tf.convert_to_tensor(np.expand_dims(bboxes.astype(np.float32), 0))
#     batch_labels = tf.convert_to_tensor(np.expand_dims(labels.astype(np.int), 0))
#     #%%
#     if i == 0:
#         _ = model((batch_imgs, batch_metas, batch_bboxes, batch_labels), training=True)
#         model.load_weights('weights/faster_rcnn0_4.h5', by_name=True)
#
#         # tf.keras.utils.plot_model(model.rpn_head, show_shapes=True, show_layer_names=True)
#     #%%
#     ########################test#################################
#     rpn_class_logits, rpn_probs = model((batch_imgs, batch_metas),training=False)
#
#     import imgTest
#
#     img1 = imgTest.showRunRpn(batch_imgs, batch_metas,rpn_class_logits, rpn_probs,100)
#     img2 = imgTest.showLabRpn(batch_imgs, batch_metas,batch_bboxes, None)
#     cv2.imshow('img1', img1)
#     cv2.imshow('img2', img2)
#     cv2.waitKey(0)

########################train#################################
# for (batch, inputs) in enumerate(train_tf_dataset):
#     batch_imgs, batch_metas, batch_bboxes, batch_labels = inputs
#     rpn_class_loss, rpn_bbox_loss, rcnn_class_loss,rcnn_bbox_loss = model((batch_imgs, batch_metas, batch_bboxes, batch_labels), training=True)
#     model.load_weights('weights/faster_rcnn3_4.h5', by_name=True)
#     break


for epoch in range(100):
    loss_history = []
    for (batch, inputs) in enumerate(train_tf_dataset):
        batch_imgs, batch_metas, batch_bboxes, batch_labels = inputs
        with tf.GradientTape() as tape:
            rpn_class_loss, rpn_bbox_loss, rcnn_class_loss,rcnn_bbox_loss = model((batch_imgs, batch_metas, batch_bboxes, batch_labels), training=True) # , rcnn_class_loss, rcnn_bbox_loss

            loss_value = rpn_class_loss  + rpn_bbox_loss*0.1  + rcnn_class_loss*0.1 + rcnn_bbox_loss*0.1

        grads = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        loss_history.append(loss_value.numpy())

        if batch % 100 == 0:
            print(rpn_class_loss, rpn_bbox_loss, rcnn_class_loss, rcnn_bbox_loss) #
            print('epoch', epoch, batch, np.mean(loss_history))
            model.save_weights('weights/faster_rcnn3_4.h5')

