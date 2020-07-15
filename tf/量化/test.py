#%%
import tensorflow as tf
import random
import numpy as np

print(tf.__version__)

saved_model_dir = './modelS'
inputs = tf.keras.Input(shape=(1),dtype=tf.float32)
X = inputs*2 + 1
X = X*5
model = tf.keras.Model(inputs, X)
tf.saved_model.save(model, saved_model_dir)

#%%
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
tflite_model = converter.convert()
open("converted_model.tflite", "wb").write(tflite_model)

#%%
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quant_model = converter.convert()
open("converted_model2.tflite", "wb").write(tflite_quant_model)
# %%

converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
def representative_dataset_gen():
    for i in range(100):
        # print(i)
        # a = tf.zeros([1],dtype=tf.float32)
        # a[0] = i/0.01
        a = np.array([[1]],dtype=np.float32)
        a[0,0] = i/0.01
        # Get sample input data as a numpy array in a method of your choosing.
        yield [a]
converter.representative_dataset = representative_dataset_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8  # or tf.uint8
converter.inference_output_type = tf.int8  # or tf.uint8
tflite_quant_model = converter.convert()
open("converted_model3.tflite", "wb").write(tflite_quant_model)
#%%


# %%


# %%


# %%
