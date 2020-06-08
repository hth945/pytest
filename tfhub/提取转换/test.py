#%%

import tensorflow as tf
import matplotlib.pyplot as plt


covn_base = tf.keras.applications.VGG16(weights='imagenet', 
                                        include_top=False, 
                                        input_shape=(150, 150, 3))
covn_base.trainable = False
covn_base.summary()

# %%
resnet50 = tf.keras.applications.ResNet50(weights='imagenet', 
                                        include_top=False, 
                                        input_shape=(224, 224, 3))
resnet50.trainable = False
resnet50.summary()


#%%
tf.keras.utils.plot_model(resnet50, show_shapes=True, show_layer_names=True)
for v in resnet50.variables:
    print(v.name, '  ', v.shape)

# %%
model = tf.saved_model.load(export_dir='../../dataAndModel/model/bit_m-r50x1_1',tags=None)

# %%
for v in model.variables:
    print(v.name, '  ', v.shape)

# %%
print(len(model.variables))
print(len(resnet50.variables))

# %%
import  datetime
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = 'logs/' + current_time
summary_writer = tf.summary.create_file_writer(log_dir) 

# %%
with summary_writer.as_default():
    tf.summary.scalar('test-acc', float(1.1), step=1)

# %%
