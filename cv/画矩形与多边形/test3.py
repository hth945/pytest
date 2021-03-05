#%%
import cv2
from cv2 import cv2
import matplotlib.pyplot as plt 
import numpy as np

plt.figure(figsize=(10, 10))
# plt.gray()

img = np.zeros([1000,1000])
print(img.shape[0])

for i in range(10):
    img[int(img.shape[0]*i/10),:]=1.0
    img[:,int(img.shape[1]*i/10)]=1.0

# img[100:200,400:450]=0.5
# img[100:200,450:500]=1.0

# img[700:800,400:450]=0.5
img[450:453,450:453]=1.0

plt.imshow(img)
plt.show()
# %%
