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

train_dataset = myDataset.myDataSet(flip_ratio=0.5,scale=(768, 768))
num_classes = len(train_dataset.get_categories())
train_generator = data_generator.DataGenerator(train_dataset)
train_tf_dataset = tf.data.Dataset.from_generator(train_generator, (tf.float32, tf.float32, tf.float32, tf.int32))
train_tf_dataset = train_tf_dataset.batch(1).prefetch(100).shuffle(100)
model = faster_rcnn2.FasterRCNN(num_classes=num_classes)

optimizer = keras.optimizers.SGD(1e-4, momentum=0.9, nesterov=True)
print([var.name for var in model.trainable_variables])


for (batch, inputs) in enumerate(train_tf_dataset):
    batch_imgs, batch_metas, batch_bboxes, batch_labels = inputs
    with tf.GradientTape() as tape:
        rpn_class_loss, rpn_bbox_loss = model((batch_imgs, batch_metas, batch_bboxes, batch_labels), training=True)
    break
print([var.name for var in model.trainable_variables])

model.load_weights('weights/faster_rcnn.h5', by_name=True)

for epoch in range(100):

    loss_history = []
    for (batch, inputs) in enumerate(train_tf_dataset):
        batch_imgs, batch_metas, batch_bboxes, batch_labels = inputs
        with tf.GradientTape() as tape:
            rpn_class_loss, rpn_bbox_loss = model((batch_imgs, batch_metas, batch_bboxes, batch_labels), training=True) # , rcnn_class_loss, rcnn_bbox_loss

            loss_value = rpn_bbox_loss # + rcnn_class_loss + rcnn_bbox_loss rpn_class_loss +

        grads = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        loss_history.append(loss_value.numpy())

        if batch % 100 == 0:
            print(rpn_class_loss, rpn_bbox_loss) # , rcnn_class_loss, rcnn_bbox_loss
            print('epoch', epoch, batch, np.mean(loss_history))
            model.save_weights('weights/faster_rcnn.h5')

