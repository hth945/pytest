#%%
import numpy as np
import random
from tensorflow.keras import models,layers,optimizers
import matplotlib.pyplot as plt
from easydict import EasyDict as edict
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf

cfg = edict()
cfg.inputSize = 28

class NpDataset(object):
    def __init__(self, dataset_type):
        self.num_batchs = 300
        self.batch_count = 0
        self.batch_size = 32

        n = 100
        x,y = np.mgrid[0.1:1:0.1,0:1:0.1]
        self.annotations = np.c_[x.ravel(), y.ravel()]
        np.random.shuffle(self.annotations)

        x,y = np.mgrid[0:1:0.01,0:1:0.01]
        self.xData = np.c_[x.ravel(), y.ravel()]
        self.yData = np.zeros([self.annotations.shape[0], 100,100])
        for i in range(self.annotations.shape[0]):
            for j in range(100):
                for k in range(100):
                    self.yData[i,j,k] = (j*0.01 -0.5)**2 * self.annotations[i,0] + (k*0.01 -0.5)**2 *self.annotations[i,1]

        self.indexX = np.array([0,5,10,15,20,24])

        i = np.array([0,5,10,15,20,27])
        j = np.array([0,5,10,15,20,27])
        X, Y = np.meshgrid(i, j)
        self.X = X.ravel()
        self.Y = Y.ravel()
        self.num_samples = self.annotations.shape[0]

    def __len__(self):
        return self.num_batchs

    def __iter__(self):
        self.batch_count = 0
        np.random.shuffle(self.yData)
        return self  

    def __next__(self):
        if self.batch_count < self.num_batchs:
            batch_src = np.zeros([self.batch_size, 36], dtype=np.float32)
            batch_labele = np.zeros([self.batch_size, cfg.inputSize,cfg.inputSize], dtype=np.float32)
            for num in range(self.batch_size):
                index = self.batch_count * self.batch_size + num
                index %= self.num_samples
                start1 = random.randint(0,100-cfg.inputSize)
                start2 = random.randint(0,100-cfg.inputSize)
                batch_labele[num,:,:] = self.yData[index,start1:start1+cfg.inputSize,start2:start2+cfg.inputSize]

                batch_src[num,:] = batch_labele[num,self.X,self.Y]

            self.batch_count += 1
            return batch_src, batch_labele
        else:
            raise StopIteration
train_dateset = NpDataset('train')

#%%
# X = np.linspace(0,1,28)
# x,y = np.meshgrid(X,X)
# lin = tf.constant([x.ravel(), y.ravel()],dtype = tf.float32)
# lin = tf.transpose(lin, [1,0])
# lin = tf.tile(lin, (train_dateset.batch_size, 1))
# print(lin.shape)
#
# for i, (src, lable) in enumerate(train_dateset):
#     lable2 = tf.reshape(lable, [-1,28*28])
#     temMax = tf.expand_dims(tf.reduce_max(src, 1), 1)
#     temMin = tf.expand_dims(tf.reduce_min(src, 1), 1)
#     print(temMax)
#     print(temMin)
#     src2 = (src - temMin) / (temMax - temMin)
#     lable2 = (lable2 - temMin) / (temMax - temMin)
#
#     src2 = tf.tile(src2,[1,28*28])
#     src2 = tf.reshape(src2, [-1, 6*6])
#     src2 = tf.concat([src2,lin], 1)
#     lable2 = tf.reshape(lable2, [-1,1])
#
#     X, Y = np.mgrid[0:cfg.inputSize:1, 0:cfg.inputSize:1]
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.scatter(X.ravel(), Y.ravel(), lable2[2*28*28:3*28*28].numpy().ravel())
#     ax.scatter(train_dateset.X, train_dateset.Y,  src2[2*28*28, :36].numpy().ravel())
#     # print(lable2[2*28*28:3*28*28].numpy().ravel()- lable[2].ravel())
#     # print(src2[2*28*28, :36].numpy().ravel() - src[2].ravel())
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.scatter(X.ravel(), Y.ravel(), lable[2].ravel())
#     ax.scatter(train_dateset.X, train_dateset.Y, src[2].ravel())
#     plt.show()
#
#     if i > 3:
#         break

#%%

import tensorflow as tf

tf.keras.backend.clear_session()


def layers(inputs,units,activation=None):
    l = tf.keras.layers.Dense(units,activation=activation)(inputs)
    x = tf.concat([inputs, l], 1)
    return l, x


inputs = tf.keras.Input(shape=(38))
_,x = layers(inputs,20, tf.nn.leaky_relu)
_,x = layers(x,20, tf.nn.leaky_relu)
_,x = layers(x,20, tf.nn.leaky_relu)
_,x = layers(x,20, tf.nn.leaky_relu)
_,x = layers(x,20, tf.nn.leaky_relu)
_,x = layers(x,20, tf.nn.leaky_relu)
_,x = layers(x,20, tf.nn.leaky_relu)
x1 = tf.keras.layers.Dense(1)(x)
model = tf.keras.Model(inputs, x1)
tf.keras.utils.plot_model(model, show_shapes=True, show_layer_names=True)
#%%
optimizer = tf.keras.optimizers.Adam(0.0002)
global_steps = 0

X = np.linspace(0,1,28)
x,y = np.meshgrid(X,X)
lin = tf.constant([x.ravel(), y.ravel()],dtype = tf.float32)
lin = tf.transpose(lin, [1,0])
lin = tf.tile(lin, (train_dateset.batch_size, 1))

for epoch in range(10):
    for src, lable in train_dateset:
        with tf.GradientTape() as tape:
            lable2 = tf.reshape(lable, [-1, 28 * 28])
            temMax = tf.expand_dims(tf.reduce_max(src, 1), 1)
            temMin = tf.expand_dims(tf.reduce_min(src, 1), 1)
            src2 = (src - temMin) / (temMax - temMin)
            lable2 = (lable2 - temMin) / (temMax - temMin)

            src2 = tf.tile(src2, [1, 28 * 28])
            src2 = tf.reshape(src2, [-1, 6 * 6])
            src2 = tf.concat([src2, lin], 1)
            lable2 = tf.reshape(lable2, [-1, 1])
            lab = model(src2, training=True)

            loss = tf.reduce_mean(tf.square(lable2 - lab))
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        global_steps += 1
        tf.print(global_steps, loss)
    tf.print("save model")
    model.save("oldModel.h5")

#%%
# model.load_weights("oldModel.h5")
X = np.linspace(0,1,28)
x,y = np.meshgrid(X,X)
lin = tf.constant([x.ravel(), y.ravel()],dtype = tf.float32)
lin = tf.transpose(lin, [1,0])
lin = tf.tile(lin, (train_dateset.batch_size, 1))
print(lin.shape)

for i, (src, lable) in enumerate(train_dateset):
    lable2 = tf.reshape(lable, [-1, 28 * 28])
    temMax = tf.expand_dims(tf.reduce_max(src, 1), 1)
    temMin = tf.expand_dims(tf.reduce_min(src, 1), 1)
    src2 = (src - temMin) / (temMax - temMin)
    lable2 = (lable2 - temMin) / (temMax - temMin)

    src2 = tf.tile(src2,[1,28*28])
    src2 = tf.reshape(src2, [-1, 6*6])
    src2 = tf.concat([src2,lin], 1)
    lable2 = tf.reshape(lable2, [-1,1])
    
    lab = model(src2, training=True)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X,Y = np.mgrid[0:cfg.inputSize:1,0:cfg.inputSize:1]
    tem = np.linspace(0,cfg.inputSize,6)
    X2,Y2 = np.meshgrid(tem,tem)
    ax.scatter(X.ravel(), Y.ravel(), lab[2*28*28:3*28*28].numpy().ravel())
    ax.scatter(X.ravel(), Y.ravel(), lable2[2*28*28:3*28*28].numpy().ravel())
    ax.scatter(X2.ravel(), Y2.ravel(), src2[2*28*28,:36].numpy().ravel())
    plt.show()

    if i > 3:
        break

# %%


