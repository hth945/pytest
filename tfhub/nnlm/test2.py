
#%%

import tensorflow as tf
import tensorflow_hub as hub
#%%
hub_url = "_"
# embed = hub.KerasLayer(hub_url,input_shape=[224*224*3])
embed = hub.KerasLayer(hub_url)
model = tf.keras.Sequential([
    embed,
])
model.build((None,)+(224,224)+(3,))
model.summary()
#%%



#%%
ws = embed.get_weights()

len(ws)
for w in ws:
    print(w.shape)
    print(w.name)
# %%
imported = tf.saved_model.load(hub_url)

# %%
for v in imported.trainable_variables:
    print(v.name, '  ', v.shape)
print('')
for v in imported.trainable_variable_ids:
    print(v.name, '  ', v.shape)
    
# for v in imported.variables:
#     print(v.name, '  ', v.shape)
# %%
print(" has {} trainable variables: {}, ...".format(
          len(imported.trainable_variables),
          ", ".join([v.name for v in imported.trainable_variables[:5]])))

# %%
tf.keras.utils.plot_model(imported, show_shapes=True, show_layer_names=True)

# %%
model = tf.keras.models.load_model(hub_url)

# %%
imported = tf.saved_model.load("_")
for v in imported.trainable_variables:
    print(v.name, '  ', v.shape)
print('')
for v in imported.trainable_variable_ids:
    print(v.name, '  ', v.shape)

# %%
import numpy as np
a = tf.zeros([1,224,224,3])
# a = tf.zeros([224,224,3])
o = imported(a)
o.shape
# %%
