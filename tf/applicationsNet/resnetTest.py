#%%
import os
import time
import shutil
import numpy as np
import tensorflow as tf

#%%
scal = 224
sampleModel = tf.keras.applications.ResNet50V2(weights='imagenet',
                                          include_top=True,
                                          input_shape=(scal, scal, 3))
sampleModel.trianable = False


tf.keras.utils.plot_model(sampleModel, to_file='ResNet50V2.png',show_shapes=True, show_layer_names=True)


# %%
scal = 224
sampleModel = tf.keras.applications.Xception(weights='imagenet',
                                          include_top=False,
                                          input_shape=(scal, scal, 3))
sampleModel.trianable = False


tf.keras.utils.plot_model(sampleModel, to_file='Xception.png',show_shapes=True, show_layer_names=True)


# %%

scal = 224
sampleModel = tf.keras.applications.MobileNetV2(weights='imagenet',
                                          include_top=False,
                                          input_shape=(scal, scal, 3))
sampleModel.trianable = False


tf.keras.utils.plot_model(sampleModel, to_file='MobileNetV2.png',show_shapes=True, show_layer_names=True)


# %%

scal = 224
sampleModel = tf.keras.applications.NASNetMobile(weights='imagenet',
                                          include_top=False,
                                          input_shape=(scal, scal, 3))
sampleModel.trianable = False


tf.keras.utils.plot_model(sampleModel, to_file='NASNetMobile.png',show_shapes=True, show_layer_names=True)



# %%

scal = 224
sampleModel = tf.keras.applications.DenseNet201(weights='imagenet',
                                          include_top=False,
                                          input_shape=(scal, scal, 3))
sampleModel.trianable = False


tf.keras.utils.plot_model(sampleModel, to_file='DenseNet201.png',show_shapes=True, show_layer_names=True)


# %%

scal = 224
sampleModel = tf.keras.applications.DenseNet121(weights='imagenet',
                                          include_top=False,
                                          input_shape=(scal, scal, 3))
sampleModel.trianable = False


tf.keras.utils.plot_model(sampleModel, to_file='DenseNet121.png',show_shapes=True, show_layer_names=True)


# %%

scal = 224
sampleModel = tf.keras.applications.InceptionResNetV2(weights='imagenet',
                                          include_top=False,
                                          input_shape=(scal, scal, 3))
sampleModel.trianable = False


tf.keras.utils.plot_model(sampleModel, to_file='InceptionResNetV2.png',show_shapes=True, show_layer_names=True)



# %%

scal = 224
sampleModel = tf.keras.applications.InceptionV3(weights='imagenet',
                                          include_top=False,
                                          input_shape=(scal, scal, 3))
sampleModel.trianable = False


tf.keras.utils.plot_model(sampleModel, to_file='InceptionV3.png',show_shapes=True, show_layer_names=True)


# %%
sampleModel.summary()



# %%
