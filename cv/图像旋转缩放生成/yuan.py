#%%
import cv2
from cv2 import cv2
import matplotlib.pyplot as plt 
import numpy as np
import random

SIZE = 20
imgSrc = cv2.imread("imgSrc.bmp", cv2.IMREAD_GRAYSCALE)
pointsSrc = np.array([[1,6],[1,22],[9,6],[9,22]])

blank = np.zeros(shape=[SIZE, SIZE]) 
for i in range(5):  
    cv2.circle(blank, (int(SIZE/2), int(SIZE/2)), i, (255-i*40), 1)
plt.imshow(blank)

# %%
