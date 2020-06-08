#%%

import tensorflow as tf
import tensorflow_hub as hub

hub_url = "https://tfhub.dev/google/tf2-preview/nnlm-en-dim128/1"

embed = hub.KerasLayer(hub_url)
embeddings = embed(["A long sentence.", "single-word", "http://example.com"])
print(embeddings.shape, embeddings.dtype)

# %%
model = tf.keras.Sequential([
    embed,
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid"),
])
tf.keras.utils.plot_model(embed, show_shapes=True, show_layer_names=True)

# %%
dict(rescale=1./255, validation_split=.20)

# %%
