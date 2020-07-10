#%%
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import datetime

def _parse_example(rand): # 将 TFRecord 文件中的每一个序列化的 tf.train.Example 解码
    parsed = { # 定义Feature结构，告诉解码器每个Feature的类型是什么
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.string)
    }
    feature_dict = tf.io.parse_single_example(rand, parsed)
    image = tf.io.decode_raw(feature_dict["image"],tf.uint8)
    image = tf.reshape(image,[512, 512, 3])
    label = tf.io.decode_raw(feature_dict["label"],tf.uint8)
    label = tf.reshape(label,[512, 512, 1])

    image = tf.cast(image, tf.float32)/127.5 - 1  # -1~1

    return image, label

BATCH_SIZE = 8
BUFFER_SIZE = 100
STEPS_PER_EPOCH = len(os.listdir('data/train/annotation')) // BATCH_SIZE

train_dataset = tf.data.TFRecordDataset('train.tfrecords')    # 读取 TFRecord 文件
train_dataset = train_dataset.map(_parse_example)
train_dataset = train_dataset.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

test_dataset = tf.data.TFRecordDataset('val.tfrecords')    # 读取 TFRecord 文件
test_dataset = test_dataset.map(_parse_example)
test_dataset = test_dataset.batch(4)

# img,mask = next(iter(train_dataset))
# plt.imshow(img[0])
# plt.show()
# plt.imshow(mask[0,:,:,0])
# plt.show()
# print(np.unique(img.numpy()))
# print(np.unique(mask.numpy()))

#%%
covn_base = tf.keras.applications.VGG16(weights='imagenet', 
                                        input_shape=(512, 512, 3),
                                        include_top=False)
layer_names = [
    'block5_conv3',   # 14x14
    'block4_conv3',   # 28x28
    'block3_conv3',   # 56x56
    'block5_pool',
]
layers = [covn_base.get_layer(name).output for name in layer_names]
# 创建特征提取模型
down_stack = tf.keras.Model(inputs=covn_base.input, outputs=layers)
down_stack.trainable = False

inputs = tf.keras.layers.Input(shape=(512, 512, 3))
o1, o2, o3, x = down_stack(inputs)
x1 = tf.keras.layers.Conv2DTranspose(512, 3, padding='same', 
                                     strides=2, activation='relu')(x)  # 14*14
x1 = tf.keras.layers.Conv2D(512, 3, padding='same', activation='relu')(x1)  # 14*14
c1 = tf.add(o1, x1)    # 14*14
x2 = tf.keras.layers.Conv2DTranspose(512, 3, padding='same', 
                                     strides=2, activation='relu')(c1)  # 14*14
x2 = tf.keras.layers.Conv2D(512, 3, padding='same', activation='relu')(x2)  # 14*14
c2 = tf.add(o2, x2)
x3 = tf.keras.layers.Conv2DTranspose(256, 3, padding='same', 
                                     strides=2, activation='relu')(c2)  # 14*14
x3 = tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu')(x3)  # 14*14
c3 = tf.add(o3, x3)

x4 = tf.keras.layers.Conv2DTranspose(128, 3, padding='same', 
                                     strides=2, activation='relu')(c3)  # 14*14
x4 = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu')(x4)  # 14*14

predictions = tf.keras.layers.Conv2DTranspose(3, 3, padding='same', 
                                     strides=2, activation='softmax')(x4)

model = tf.keras.models.Model(inputs=inputs, outputs=predictions)
model.summary()
#%%
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
history = model.fit(train_dataset, 
                          epochs=20,
                          steps_per_epoch=STEPS_PER_EPOCH,
                          validation_steps=1,
                          validation_data=test_dataset)


# %%
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(32)

plt.plot(history.epoch, history.history['loss'], 'r', label='loss')
plt.plot(history.epoch, history.history['val_loss'], 'b--', label='val_loss')
plt.legend()

#%%
num = 3
for image, mask in test_dataset.take(1):
    pred_mask = model.predict(image)
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    
    plt.figure(figsize=(10, 10))
    for i in range(num):
        plt.subplot(num, 3, i*num+1)
        plt.imshow(tf.keras.preprocessing.image.array_to_img(image[i]))
        plt.subplot(num, 3, i*num+2)
        plt.imshow(tf.keras.preprocessing.image.array_to_img(mask[i]))
        plt.subplot(num, 3, i*num+3)
        plt.imshow(tf.keras.preprocessing.image.array_to_img(pred_mask[i]))

for image, mask in train_dataset.take(1):
    pred_mask = model.predict(image)
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    
    plt.figure(figsize=(10, 10))
    for i in range(num):
        plt.subplot(num, 3, i*num+1)
        plt.imshow(tf.keras.preprocessing.image.array_to_img(image[i]))
        plt.subplot(num, 3, i*num+2)
        plt.imshow(tf.keras.preprocessing.image.array_to_img(mask[i]))
        plt.subplot(num, 3, i*num+3)
        plt.imshow(tf.keras.preprocessing.image.array_to_img(pred_mask[i]))
#%%
model.save('fcn.h5')
