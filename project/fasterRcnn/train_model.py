#%%
import  os
import  tensorflow as tf
from    tensorflow import keras
import  numpy as np
from    matplotlib import pyplot as plt
from    detection.datasets import myDataset,data_generator
from    detection.models import faster_rcnn

tf.random.set_seed(22)
np.random.seed(22)

train_dataset = myDataset.myDataSet(flip_ratio=0.5,scale=(800, 1216))
num_classes = len(train_dataset.get_categories())
train_generator = data_generator.DataGenerator(train_dataset)
train_tf_dataset = tf.data.Dataset.from_generator(train_generator, (tf.float32, tf.float32, tf.float32, tf.int32))
train_tf_dataset = train_tf_dataset.batch(1).prefetch(100).shuffle(100)
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

    break

model.load_weights('weights/faster_rcnn.h5', by_name=True)
#     exit()

# %%

for epoch in range(100):

    loss_history = []
    for (batch, inputs) in enumerate(train_tf_dataset):
        batch_imgs, batch_metas, batch_bboxes, batch_labels = inputs
        with tf.GradientTape() as tape:
            rpn_class_loss, rpn_bbox_loss, rcnn_class_loss, rcnn_bbox_loss = model((batch_imgs, batch_metas, batch_bboxes, batch_labels), training=True)

            loss_value = rpn_class_loss + rpn_bbox_loss + rcnn_class_loss + rcnn_bbox_loss

        grads = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        loss_history.append(loss_value.numpy())

        if batch % 100 == 0:
            print('epoch', epoch, batch, np.mean(loss_history))
            #model.save_weights('weights/faster_rcnn.h5')