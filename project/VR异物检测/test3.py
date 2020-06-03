import cv2
from config import cfg
import numpy as np
import tensorflow as tf
import datetime
import os
import xml.etree.ElementTree as ET
import csv

# img = cv2.imread('../newLable/Image_20200416200723024.bmp')

modelTest = tf.keras.models.load_model('./frozen_models')
# modelTest.train = False
@tf.function
def runModel(imgTem):
    imgTem = tf.cast(imgTem,dtype=tf.float32)
    return modelTest(imgTem)

print(modelTest.summary())


