import cv2
from cv2 import cv2
import matplotlib.pyplot as plt
import numpy as np
import random

SIZE = 40
mask = np.zeros([SIZE, SIZE],dtype=np.uint8)
contours = np.array([[1,1],[1,10],[20,10],[20,1]])

cv2.fillPoly(mask, pts=[contours], color=(255))  # 内部为1
cv2.polylines(mask, pts=[contours], isClosed=True, color=64, thickness=3) # 边界为0.5

cv2.imshow('msk', mask)

mask = np.zeros([SIZE, SIZE],dtype=np.uint8)
mask[1:10,1:20] = 64
cv2.imshow('msk2', mask)


mask = np.zeros([SIZE, SIZE],dtype=np.uint8)
cv2.rectangle(mask , (1, 1), (20, 10), (255), -1, 4)
cv2.imshow('msk3', mask)

cv2.waitKey(0)


