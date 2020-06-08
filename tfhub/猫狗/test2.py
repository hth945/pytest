#%%
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import glob
import os

train_image_path = glob.glob('../../dataAndModel/data/dc_2000/train/*/*.jpg')
train_image_path = train_image_path[0:2000:10]
train_image_label = [int(p.split('\\')[1] == 'cat') for p in train_image_path]
# print(train_image_path)
# exit()
# %%
def load_preprosess_image(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [224, 224])
    image = tf.cast(image, tf.float32)
    image = image/255
    return image, label

BATCH_SIZE = 1
train_count = len(train_image_path)

train_image_ds = tf.data.Dataset.from_tensor_slices((train_image_path, train_image_label))
train_image_ds = train_image_ds.map(load_preprosess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
# for img, label in train_image_ds.take(2):
#     plt.imshow(img)
# plt.show()
train_image_ds = train_image_ds.shuffle(train_count).repeat().batch(BATCH_SIZE)


test_image_path = glob.glob('../../dataAndModel/data/dc_2000/test/*/*.jpg')
test_image_label = [int(p.split('\\')[1] == 'cat') for p in test_image_path]
test_image_ds = tf.data.Dataset.from_tensor_slices((test_image_path, test_image_label))
test_image_ds = test_image_ds.map(load_preprosess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
test_image_ds = test_image_ds.repeat().batch(BATCH_SIZE)
test_count = len(test_image_path)

# %%
import tensorflow_hub as hub
resnet50 = tf.saved_model.load('../../dataAndModel/model/bit_m-r50x1_1')
inputs = tf.keras.Input(shape=(224, 224,3))
x = resnet50(inputs)
x = tf.keras.layers.Dense(512, activation='relu')(x)
x = tf.keras.layers.Dense(1, activation='sigmoid')(x)
model = tf.keras.Model(inputs, x)
model.summary()
# model = tf.keras.models.load_model('oldModel.h5',custom_objects={'KerasLayer':hub.KerasLayer})
# model.summary()
# tf.keras.utils.plot_model(model, show_shapes=True, show_layer_names=True)
model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0005),
              loss='binary_crossentropy',
              metrics=['acc'])

history = model.fit(
    train_image_ds,
    steps_per_epoch=train_count//BATCH_SIZE,
    epochs=3,
    validation_data=test_image_ds,
    validation_steps=test_count//BATCH_SIZE)

model.save("oldModel.h5")