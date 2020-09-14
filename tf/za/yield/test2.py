#%%
import numpy as np
import tensorflow as tf

class myDataSet(object):
    def __init__(self,):
        self.data1 = [np.ones([10]) for i in range(10)]
        self.data2 = [np.ones([10]) for i in range(10)]
        

    def __len__(self):
        return len(self.data1)

    def __getitem__(self, idx):
        return self.data1[idx], self.data2[idx]


class DataGenerator:

    def __init__(self, dataset, shuffle=False):
        self.dataset = dataset
        self.shuffle = shuffle
    
    def __call__(self):
        indices = np.arange(len(self.dataset))
        if self.shuffle:
            np.random.shuffle(indices)
        for img_idx in indices:
            # img, img_meta, bbox, label = self.dataset[img_idx]
            # yield img, img_meta, bbox, label
            img, img_meta = self.dataset[img_idx]
            yield img , img_meta
#%%
train_dataset = myDataSet()
data_generator = DataGenerator(train_dataset)
for i in data_generator():
    print(i)

#%%
dataset = tf.data.Dataset.from_generator(data_generator, (tf.int32,tf.int32))

for i in dataset:
    print(i)
    break
#%%
dataset = tf.data.Dataset.from_generator(data_generator, (tf.int32,tf.int32)).batch(2)
for i in dataset:
    print(i)
    break
# %%
train_dataset[0]
# %%
len(train_dataset)
# %%
data_generator().next()
# %%
