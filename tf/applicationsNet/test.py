#%%
import os
import time
import shutil
import numpy as np
import tensorflow as tf


scal = 224
xception = tf.keras.applications.Xception(weights='imagenet',
                                          include_top=False,
                                          input_shape=(scal, scal, 3))
xception.trianable = False


# %%
tf.keras.utils.plot_model(xception, show_shapes=True, show_layer_names=True)

# %%
x = tf.keras.layers.GlobalAveragePooling2D()(xception.output)
x = tf.keras.layers.Dense(2048, activation='relu')(x)
x = tf.keras.layers.Dense(256, activation='relu')(x)
out = tf.keras.layers.Dense(3)(x)
model = tf.keras.models.Model(inputs=xception.input, outputs=out)
tf.keras.utils.plot_model(model, show_shapes=True, show_layer_names=True)


# %%
