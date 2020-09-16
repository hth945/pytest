#%%
import tensorflow as tf


class ResnetIdentityBlock(tf.keras.Model):
    def __init__(self, kernel_size, filters):
        super(ResnetIdentityBlock, self).__init__(name='')
        filters1, filters2, filters3 = filters

        self.conv2a = tf.keras.layers.Conv2D(filters1, (1, 1))
        self.bn2a = tf.keras.layers.BatchNormalization()

        self.conv2b = tf.keras.layers.Conv2D(filters2, kernel_size, padding='same')
        self.bn2b = tf.keras.layers.BatchNormalization()

        self.conv2c = tf.keras.layers.Conv2D(filters3, (1, 1))
        self.bn2c = tf.keras.layers.BatchNormalization()


    def call(self, input_tensor, training=False):
        x = self.conv2a(input_tensor)
        x = self.bn2a(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2b(x)
        x = self.bn2b(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2c(x)
        x = self.bn2c(x, training=training)

        x += input_tensor
        return tf.nn.relu(x)

    def tem(self, ):
        pass


block = ResnetIdentityBlock(1, [1, 2, 3])
block(tf.zeros([1, 2, 3, 3])) 
block.summary()

# %%

# class MyDenseLayer(tf.keras.layers.Layer):
#   def __init__(self, num_outputs):
#     super(MyDenseLayer, self).__init__()
#     self.num_outputs = num_outputs

#   def build(self, input_shape):
#     self.kernel = self.add_weight("kernel",
#                                   shape=[int(input_shape[-1]),
#                                          self.num_outputs])

#   def call(self, input):
#     return tf.matmul(input, self.kernel)

# layer = MyDenseLayer(10)
# _ = layer(tf.zeros([10, 5])) # Calling the layer `.builds` it.
# print([var.name for var in layer.trainable_variables])

# %%
class MyDenseLayer(tf.keras.Model):
    def __init__(self, num_outputs):
        super(MyDenseLayer, self).__init__()
        self.num_outputs = num_outputs
        self.kernel = self.add_weight("kernel",shape=[5,self.num_outputs])

    def call(self, input):
        input *= 2
        t = [self.testList(i) for i in range(input.shape[0])]
        if len(t):
            return 1
        else:
            return 0
        
        return t

    def testList(self, i):

        return i


layer = MyDenseLayer(10)
layer(tf.zeros([0, 5])) 

# %%
layer.summary()
# %%
print([var.name for var in layer.trainable_variables])
# %%
layer(tf.zeros([1, 5])) 
# %%
