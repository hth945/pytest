#%%
import cv2
from cv2 import cv2
import matplotlib.pyplot as plt 
import numpy as np

imgName = "t.jpg"
imgSrc = cv2.imread(imgName, cv2.IMREAD_GRAYSCALE)
image = cv2.resize(imgSrc, (128,  76),interpolation=cv2.INTER_AREA)
imgSrc = cv2.imread(imgName)
imgSrc = imgSrc[...,[2,0,1]]

N = 15
plt.figure(figsize=(N, N))
plt.imshow(imgSrc)
plt.show()
plt.figure(figsize=(N, N))
plt.imshow(image, cmap='gray')
plt.show()

image = cv2.resize(image, (100,  100),interpolation=cv2.INTER_AREA)
plt.figure(figsize=(N, N))
plt.imshow(image, cmap='gray')
plt.show()

# %%
imgSrc = cv2.imread(imgName)
imgSrc2 = imgSrc[...,[2,1,0]]
N = 15
plt.figure(figsize=(N, N))
plt.imshow(imgSrc2)
plt.show()
# %%
imgSrc = cv2.imread(imgName, cv2.IMREAD_GRAYSCALE)
image = cv2.resize(imgSrc, (32,  24),interpolation=cv2.INTER_AREA)
N = 15
plt.figure(figsize=(N, N))
plt.imshow(image, cmap='gray')
plt.show()
# %%

mask = np.zeros([24, 32],dtype=np.uint8)
contours = np.array([[1,1],[1,10],[20,10],[20,1]])

cv2.fillPoly(mask, pts=[contours], color=(255))  # 内部为1
cv2.polylines(mask, pts=[contours], isClosed=True, color=64, thickness=3) # 边界为0.5

N = 15
plt.figure(figsize=(N, N))
plt.imshow(mask, cmap='gray')
plt.show()








# %%
