import tensorflow as tf
from tensorflow.keras import layers

class myTestNet(tf.keras.Model):

    def __init__(self, **kwargs):
        super(myTestNet, self).__init__(**kwargs)

        self.bn_conv0 = layers.BatchNormalization(name='bn_conv0')
        self.conv0a = layers.Conv2D(16, (3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal',
                                    name='conv0a')
        self.conv0b = layers.Conv2D(16, (3, 3), strides=(2, 2), padding='same', kernel_initializer='he_normal',
                                    name='conv0b')
        self.conv0c = layers.Conv2D(16, (3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal',
                                    name='conv0c')

        self.bn_conv1 = layers.BatchNormalization(name='bn_conv1')
        self.conv1a = layers.Conv2D(32, (3, 3), strides=(2, 2), padding='same', kernel_initializer='he_normal', name='conv1a')
        self.conv1b = layers.Conv2D(32, (3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal',name= 'conv1b')

        self.bn_conv2 = layers.BatchNormalization(name='bn_conv2')
        self.conv2a = layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same', kernel_initializer='he_normal',
                                    name='conv2a')
        self.conv2b = layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal',
                                    name='conv2b')

        self.bn_conv3 = layers.BatchNormalization(name='bn_conv3')
        self.conv3a = layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same', kernel_initializer='he_normal',
                                    name='conv3a')
        self.conv3b = layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal',
                                    name='conv3b')

        self.bn_conv4 = layers.BatchNormalization(name='bn_conv4')
        self.conv4a = layers.Conv2D(256, (3, 3), strides=(2, 2), padding='same', kernel_initializer='he_normal',
                                    name='conv4a')
        self.conv4b = layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal',
                                    name='conv4b')

        self.out_channel = (32, 64, 128, 256)

    def call(self, inputs, training=True):
        x = self.conv0a(inputs, training=training)
        x = tf.nn.relu(x)
        x = self.bn_conv0(x, training=training)
        x = self.conv0b(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv0c(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv1a(x, training=training)
        x = tf.nn.relu(x)
        x = self.bn_conv1(x, training=training)
        x = self.conv1b(x, training=training)
        C2 = x = tf.nn.relu(x)

        x = self.conv2a(x, training=training)
        x = tf.nn.relu(x)
        x = self.bn_conv2(x, training=training)
        x = self.conv2b(x, training=training)
        C3 = x = tf.nn.relu(x)

        x = self.conv3a(x, training=training)
        x = tf.nn.relu(x)
        x = self.bn_conv3(x, training=training)
        x = self.conv3b(x, training=training)
        C4 = x = tf.nn.relu(x)

        x = self.conv4a(x, training=training)
        x = tf.nn.relu(x)
        x = self.bn_conv4(x, training=training)
        x = self.conv4b(x, training=training)
        C5 = x = tf.nn.relu(x)
        return (C2, C3, C4, C5)

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        batch, H, W, C = shape

        C2_shape = tf.TensorShape([batch, H // 4, W // 4, self.out_channel[0]])
        C3_shape = tf.TensorShape([batch, H // 8, W // 8, self.out_channel[1]])
        C4_shape = tf.TensorShape([batch, H // 16, W // 16, self.out_channel[2]])
        C5_shape = tf.TensorShape([batch, H // 32, W // 32, self.out_channel[3]])

        return (C2_shape, C3_shape, C4_shape, C5_shape)


# %%
if __name__ == '__main__':
    random_float = tf.random.uniform(shape=(10, 768, 768, 3))
    backbone = myTestNet(name='res_net')

    C2, C3, C4, C5 = backbone(random_float)
    print(C2.shape)
    print(C3.shape)
    print(C4.shape)
    print(C5.shape)
    print(backbone.compute_output_shape((1, 768, 768, 3)))

# %%
