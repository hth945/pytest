#%%
import  os
import  numpy as np
from    matplotlib import pyplot as plt
import cv2
from    detection.datasets import myDataset,data_generator
from    detection.models import faster_rcnn
import  tensorflow as tf
from    tensorflow import keras

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

tf.random.set_seed(22)
np.random.seed(22)

train_dataset = myDataset.myDataSet(flip_ratio=0.5,scale=(768, 768))


num_classes = len(train_dataset.get_categories())
train_generator = data_generator.DataGenerator(train_dataset)
train_tf_dataset = tf.data.Dataset.from_generator(train_generator, (tf.float32, tf.float32, tf.float32, tf.int32))
train_tf_dataset = train_tf_dataset.batch(2).prefetch(100).shuffle(100)
# train_tf_dataset = train_tf_dataset.batch(3)

# %%
model = faster_rcnn.FasterRCNN(num_classes=num_classes)

optimizer = keras.optimizers.SGD(1e-3, momentum=0.9, nesterov=True)

for (batch, inputs) in enumerate(train_tf_dataset):
    batch_imgs, batch_metas, batch_bboxes, batch_labels = inputs
    print(batch_imgs.shape)
    print(batch_metas)
    print(batch_bboxes)
    print(batch_labels)

    print('runMOdel')
    _ = model((batch_imgs, batch_metas), training=False)
    model.load_weights('weights/faster_rcnn.h5', by_name=True)
    break

# for (batch, inputs) in enumerate(train_tf_dataset):
#     batch_imgs, batch_metas, batch_bboxes, batch_labels = inputs
#     o,rois_list, rcnn_target_matchs_list, rcnn_target_deltas_list = model((batch_imgs, batch_metas, batch_bboxes, batch_labels), training=True)
#
#     image = batch_imgs[0].numpy()
#     # print(rois_list)
#     print(o[0], o[1])
#     bboxs = rois_list[0].numpy()
#     for i in range(bboxs.shape[0]):
#         bbox = bboxs[i]*1216
#         image = cv2.rectangle(image, (int(float(bbox[0])),
#                                       int(float(bbox[1]))),
#                                       (int(float(bbox[2])),
#                                        int(float(bbox[3]))), (255, 0, 0), 2)
#
#     cv2.imshow('img', image)
#     cv2.waitKey(0)




# %%

for epoch in range(100):

    loss_history = []
    for (batch, inputs) in enumerate(train_tf_dataset):
        batch_imgs, batch_metas, batch_bboxes, batch_labels = inputs
        with tf.GradientTape() as tape:
            rpn_class_loss, rpn_bbox_loss, rcnn_class_loss, rcnn_bbox_loss = model((batch_imgs, batch_metas, batch_bboxes, batch_labels), training=True)

            loss_value = rpn_class_loss + rpn_bbox_loss # + rcnn_class_loss + rcnn_bbox_loss

        grads = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        loss_history.append(loss_value.numpy())

        if batch % 100 == 0:
            print(rpn_class_loss, rpn_bbox_loss, rcnn_class_loss, rcnn_bbox_loss )
            print('epoch', epoch, batch, np.mean(loss_history))
            model.save_weights('weights/faster_rcnn.h5')

