#%%

import tensorflow as tf
import tensorflow_hub as hub

class CustomModule(tf.Module):

  def __init__(self):
    super(CustomModule, self).__init__()
    self.v = tf.Variable(1.)

  @tf.function
  def __call__(self, x):
    return x * self.v

  @tf.function(input_signature=[tf.TensorSpec([], tf.float32)])
  def mutate(self, new_v):
    self.v.assign(new_v)

module = CustomModule()
module(tf.constant(0.))
#%%

tf.saved_model.save(module, "tmp/module_no_signatures")


# %%
imported = tf.saved_model.load("tmp/module_no_signatures")
assert 3. == imported(tf.constant(3.)).numpy()
imported.mutate(tf.constant(2.))
assert 6. == imported(tf.constant(3.)).numpy()


# %%
