#%%
import numpy as np 
import tensorflow as tf 
from tensorflow import keras 
from matplotlib import pyplot as plt 
import xml.etree.ElementTree as ET
import cv2 
# %%
# ann_dir = 'data/train/annotation'
# img_dir = 'data/train/image'
# tfrecord_file = 'train.tfrecords'

ann_dir = 'data/val/annotation'
img_dir = 'data/val/image'
tfrecord_file = 'val.tfrecords'
obj_names = ('sugarbeet', 'weed')

print(os.listdir(ann_dir)[-5:])
print(os.listdir(img_dir)[-5:])
len(os.listdir('data/train/annotation'))

# # %%
# tree = ET.parse(os.path.join(ann_dir, 'X-10-0.xml'))
# mask = np.zeros([512,512,1],dtype=np.uint8)
# for elem in tree.iter():
#     if 'filename' == elem.tag:
#         imgP = os.path.join(img_dir, elem.text)
#         img = cv2.imread(imgP)
        
#     if 'object' == elem.tag:
#         xmin = int(elem.find('bndbox/xmin').text)
#         ymin = int(elem.find('bndbox/ymin').text)
#         xmax = int(elem.find('bndbox/xmax').text)
#         ymax = int(elem.find('bndbox/ymax').text)
#         label = obj_names.index(elem.find('name').text) + 1
#         mask[ymin: ymax, xmin:xmax, 0] = label
#         # cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 255), 2)
# plt.imshow(img)
# plt.show()
# plt.imshow(mask[:,:,0])
# plt.show()

# %%
with tf.io.TFRecordWriter(tfrecord_file) as writer:
    for anno in os.listdir(ann_dir):
        tree = ET.parse(os.path.join(ann_dir, anno))
        mask = np.zeros([512,512,1],dtype=np.uint8)
        for elem in tree.iter():
            if 'filename' == elem.tag:
                imgP = os.path.join(img_dir, elem.text)
                img = cv2.imread(imgP)
                
            if 'object' == elem.tag:
                xmin = int(elem.find('bndbox/xmin').text)
                ymin = int(elem.find('bndbox/ymin').text)
                xmax = int(elem.find('bndbox/xmax').text)
                ymax = int(elem.find('bndbox/ymax').text)
                label = obj_names.index(elem.find('name').text) + 1
                mask[ymin: ymax, xmin:xmax, 0] = label
                # cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 255), 2)
        feature = {                             # 建立 tf.train.Feature 字典
            'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img.tostring()])),  # 图片是一个 Bytes 对象
            'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[mask.tostring()]))   # 标签是一个 Int 对象
        }
        example = tf.train.Example(features=tf.train.Features(feature=feature)) # 通过字典建立 Example
        writer.write(example.SerializeToString())   # 将Example序列化并写入 TFRecord 文件

# %%
def _parse_example(rand): # 将 TFRecord 文件中的每一个序列化的 tf.train.Example 解码
    parsed = { # 定义Feature结构，告诉解码器每个Feature的类型是什么
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.string)
    }
    feature_dict = tf.io.parse_single_example(rand, parsed)
    image = tf.io.decode_raw(feature_dict["image"],tf.uint8)
    image = tf.reshape(image,[512, 512, 3])
    label = tf.io.decode_raw(feature_dict["label"],tf.uint8)
    label = tf.reshape(label,[512, 512, 1])

    return image, label

raw_dataset = tf.data.TFRecordDataset(tfrecord_file)    # 读取 TFRecord 文件
dataset = raw_dataset.map(_parse_example)
img,mask = next(iter(dataset))
plt.imshow(img)
plt.show()
plt.imshow(mask[:,:,0])
plt.show()
# %%
