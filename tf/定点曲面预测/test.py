#%%
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras import models,layers,optimizers
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from easydict import EasyDict as edict

cfg = edict()
cfg.imputSize = 28
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

        i = np.array([0,5,10,15,20,24])
        j = np.array([0,5,10,15,20,24])
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
            batch_src = np.zeros([self.batch_size, cfg.imputSize, cfg.imputSize,1], dtype=np.float32)
            batch_labele = np.zeros([self.batch_size, cfg.imputSize, cfg.imputSize,1], dtype=np.float32)
            for num in range(self.batch_size):
                index = self.batch_count * self.batch_size + num
                index %= self.num_samples
                start1 = random.randint(0,100-cfg.imputSize)
                start2 = random.randint(0,100-cfg.imputSize)
                batch_labele[num,:,:,0] = self.yData[index,start1:start1+cfg.imputSize,start2:start2+cfg.imputSize]
                batch_src[num,self.X,self.Y,0] = batch_labele[num,self.X,self.Y,0]

            self.batch_count += 1
            return batch_src, batch_labele
        else:
            raise StopIteration

# train_dateset = NpDataset('train')
# for i, (src, lable) in enumerate(train_dateset):
#     # print(src[2])
#     # print(lable[2])
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     X,Y = np.mgrid[0:cfg.imputSize:1,0:cfg.imputSize:1]
#     ax.scatter(X.ravel(), Y.ravel(), src[2].ravel())
#     plt.show()
#     # ax.scatter(X.ravel(), Y.ravel(), lable[2].ravel())
#     # plt.show() 
#     if i > 3:
#         break


#%%
tf.keras.backend.clear_session()

inputs = tf.keras.Input(shape=(cfg.imputSize,cfg.imputSize,1))
X = tf.keras.layers.Conv2D(16, (3, 3), padding='same', strides=2, activation='relu')(inputs)
X = tf.keras.layers.Conv2D(64, (3, 3), padding='same', strides=2, activation='relu')(X)

X = tf.keras.layers.UpSampling2D(2, interpolation='bilinear')(X)
X = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')(X)
X = tf.keras.layers.UpSampling2D(2, interpolation='bilinear')(X)
X = tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu')(X)

X = tf.keras.layers.Conv2D(8, (3, 3), padding='same', activation='relu')(X)
X = tf.keras.layers.Conv2D(1, (3, 3), padding='same', activation='relu')(X)

model = tf.keras.Model(inputs, X)
tf.keras.utils.plot_model(model, show_shapes=True, show_layer_names=True)

# %%

optimizer = tf.keras.optimizers.Adam(0.0002)
global_steps = 0
trainset = NpDataset('train')
for epoch in range(10):
    for image_data, target in trainset:
        with tf.GradientTape() as tape:
            lab1 = model(image_data, training=True)
            loss = tf.reduce_mean(tf.square(target - lab1))
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        global_steps += 1
        tf.print(global_steps,loss)
    tf.print("save model")
    model.save("oldModel.h5")

models = tf.keras.models.load_model('oldModel.h5')
#%%
train_dateset = NpDataset('train')
for i, (src, lable) in enumerate(train_dateset):
    # print(src[2])
    # print(lable[2])
    lab1 = model(src, training=True)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X,Y = np.mgrid[0:cfg.imputSize:1,0:cfg.imputSize:1]

    loss = tf.reduce_mean(tf.square(lable - lab1))
    print(loss)
    # ax.scatter(X.ravel(), Y.ravel(), src[2].ravel())
    # plt.show()

    ax.scatter(X.ravel(), Y.ravel(), lab1[2].numpy().ravel())
    plt.show()
    
    # ax.scatter(X.ravel(), Y.ravel(), lable[2].ravel())
    # plt.show() 
    if i > 3:
        break


# %%


