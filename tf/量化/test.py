#%%
import tensorflow as tf
import random
import numpy as np

print(tf.__version__)

# %%


saved_model_dir = './modelS'
inputs = tf.keras.Input(shape=(1))
outputs = inputs*2
model = tf.keras.Model(inputs, outputs)
tf.saved_model.save(model, saved_model_dir)

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
    for _ in range(100):
        i = np.array(random.random(),dtype=np.float32)
        # Get sample input data as a numpy array in a method of your choosing.
        yield [i]
converter.representative_dataset = representative_dataset_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8  # or tf.uint8
converter.inference_output_type = tf.int8  # or tf.uint8
tflite_quant_model = converter.convert()
#%%


# %%


# %%


# %%
