#%%
import tensorflow as tf
import tensorflow_hub as hub
import cv2
# https://tfhub.dev/google/bit/m-r50x1/1
module = hub.KerasLayer("https://tfhub.dev/google/bit/m-r50x1/imagenet21k_classification/1", output_shape=[20])
# images = ...  # A batch of images with shape [batch_size, height, width, 3].
# logits = module(images)  # Logits with shape [batch_size, 21843].
# probabilities = tf.nn.sigmoid(logits)

# %%
# module.compute_output_shape([2, 224, 224, 3])

# %%
module.get_weights()

# %%
model = tf.keras.Sequential([
  module,
])
# model.build(input_shape=(1,224,224,3))
model.summary()
# tf.keras.utils.plot_model(model)
# %%
