#%%
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras import models,layers,optimizers
import matplotlib.pyplot as plt

class NpDataset(object):
    def __init__(self, dataset_type):
        self.num_batchs = 300
        self.batch_count = 0
        self.batch_size = 320

        n = 100
        x,y = np.mgrid[0.1:1:0.1,0:1:0.1]
        self.annotations = np.c_[x.ravel(), y.ravel()]
        np.random.shuffle(self.annotations)
        self.xData = np.arange(0,n) / n
        # self.yData = np.matmul(self.annotations[:,0:1],self.xData[np.newaxis,:])  + self.annotations[:,1:]
        self.yData = np.zeros([self.annotations.shape[0], self.xData.shape[0]])
        for i in range(self.annotations.shape[0]):
            for j in range(self.xData.shape[0]):
                self.yData[i,j] = self.annotations[i,0] * self.xData[j] + self.annotations[i,1]

        self.num_samples = self.annotations.shape[0]

    def __len__(self):
        return self.num_batchs

    def __iter__(self):
        self.batch_count = 0
        np.random.shuffle(self.yData)
        return self  

    def __next__(self):
        if self.batch_count < self.num_batchs:
            batch_src = np.zeros([self.batch_size, 25], dtype=np.float32)
            batch_labele = np.zeros([self.batch_size, 25], dtype=np.float32)
            for num in range(self.batch_size):
                index = self.batch_count * self.batch_size + num
                index %= self.num_samples
                start = random.randint(0,100-25)
                
                batch_labele[num,:] = self.yData[index,start:start+25]
                # i = np.random.choice(np.arange(25), size=4, replace=False)
                i = np.array([0,5,10,15,20,24])
                batch_src[num,i] = batch_labele[num,i]

            self.batch_count += 1
            return batch_src, batch_labele
        else:
            raise StopIteration

# train_dateset = NpDataset('train')
# for src, lable in train_dateset:
#     print(src[2])
#     print(lable[2])
#     x = np.arange(0,25)
#     plt.plot(x,src[2])
#     plt.plot(x,lable[2])
#     plt.show()
#     break

#%%
tf.keras.backend.clear_session()

inputs = tf.keras.Input(shape=(25))
x = tf.keras.layers.Dense(50,activation="relu")(inputs)
x = tf.keras.layers.Dense(50,activation="relu")(x)
x1 = tf.keras.layers.Dense(25)(x)
x = tf.keras.layers.Dense(50,activation="relu")(x1)
x = tf.keras.layers.Dense(50,activation="relu")(x)
x2 = tf.keras.layers.Dense(25)(x)
model = tf.keras.Model(inputs, [x1, x2])
tf.keras.utils.plot_model(model, show_shapes=True, show_layer_names=True)

# %%

optimizer = tf.keras.optimizers.Adam(0.0002)
global_steps = 0
trainset = NpDataset('train')
for epoch in range(10):
    for image_data, target in trainset:
        with tf.GradientTape() as tape:
            lab1,lab2 = model(image_data, training=True)
            loss = tf.reduce_mean(tf.square(target - lab1)) + tf.reduce_mean(tf.square(target - lab2))
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        global_steps += 1
        tf.print(global_steps,loss)
    tf.print("save model")
    model.save("oldModel.h5")

train_dateset = NpDataset('train')
x = np.arange(0,25)
for i, (src, lable) in enumerate(train_dateset):
    print(src[2])
    print(lable[2])
    lab1,lab2 = model(src, training=True)
    print(lab1[2].numpy())
    print(lab2[2].numpy())

    # plt.plot(x,src[2])
    
    plt.plot(x,lable[2])
    plt.plot(x,lab1[2].numpy())
    plt.plot(x,lab2[2].numpy())
    plt.show()

    if i > 3:
        break

# %%


