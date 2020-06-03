import cv2
from config import cfg
import numpy as np
import tensorflow as tf
import datetime
import os

# 将程序限定在一块GPU上
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
# 限制使用2G显存
tf.config.experimental.set_virtual_device_configuration(
    gpus[0],
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024*6)]
)

img = cv2.imread('../newLable/Image_20200416200723024.bmp')

# a = np.fromfile('t\\5_5.bin',dtype=np.uint8)
# img = np.reshape(a, [3648,5472,3])
# img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
# cv2.imshow('img', img)
# cv2.waitKey(0)



modelTest = tf.keras.models.load_model('testmodel.h5')
modelTest.train = False
print(modelTest.input)
print(modelTest.output)
@tf.function
def runModel(imgTem):
    imgTem = tf.cast(imgTem,dtype=tf.float32)
    return modelTest(imgTem)

# def tfTest():
#
#     print(modelTest.input)
#     print(modelTest.output)
#     img = np.zeros([1,3648, 5472, 3],dtype=np.uint8)
#     # img2 = img.astype(np.float32)
#     print(img.dtype)
#     print(img.shape)
#
#     for i in range(100):
#         begin = datetime.datetime.now()
#         label = runModel(img)
#         end = datetime.datetime.now()
#         print(end - begin)
#
#     # print(label.shape)
# tfTest()

label = runModel(img[np.newaxis])
print(label.numpy().shape)
cv2.imshow('img', img)
cv2.imshow('label', label.numpy()[0,:,:,1])
print(img.shape)
cv2.waitKey(0)


