import cv2
from config import cfg
import numpy as np
import tensorflow as tf
import datetime


img = cv2.imread('../newlableWU/2_6.bin.png')

# a = np.fromfile('t\\5_5.bin',dtype=np.uint8)
# img = np.reshape(a, [3648,5472,3])
# img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
# cv2.imshow('img', img)
# cv2.waitKey(0)



modelTest = tf.keras.models.load_model('testmodel.h5')
modelTest.train = False
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

imgs = img / 255.0
label = runModel(imgs[np.newaxis])
print(label.numpy().shape)
cv2.imshow('img', img)
cv2.imshow('label', label.numpy()[0,:,:,0])
print(img.shape)
print("imgs max :", np.max(imgs[np.newaxis]))
print("label max :", np.max(label.numpy()[0,:,:,0]))
cv2.waitKey(0)



