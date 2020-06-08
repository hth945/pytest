#%%

import keras2onnx
from tensorflow import keras
import tensorflow as tf
import onnx

model = keras.models.load_model('fcn.h5')
print(model.name)
onnx_model = keras2onnx.convert_keras(model, model.name)
temp_model_file = 'fcn.onnx'
onnx.save_model(onnx_model, temp_model_file)

# %%
from tensorflow import keras
import tensorflow as tf

model = keras.models.load_model('fcn.h5')
tf.keras.models.save_model(model,"model_save_path")

# %%
import tensorflow as tf

saved_model_dir = 'model_save_path'
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
tflite_model = converter.convert()
open("converted_model.tflite", "wb").write(tflite_model)

# %%
import tensorflow as tf
from tensorflow import keras

model = keras.models.load_model('fcn.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
open("converted_model_h5.tflite", "wb").write(tflite_model)

# %%
