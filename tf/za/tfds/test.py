#%%
import tensorflow_datasets as tfds
# The full `train` split.
train_ds = tfds.load('omniglot', split='train')
# %%
train_ds
# %%
for i in train_ds:
    # print(i[0].shape,i[1].shape)
    print(i['label'].shape)
    print(i['image'].shape)
    
    break
# %%
