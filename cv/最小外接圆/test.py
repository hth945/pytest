#%%
import cv2
from cv2 import cv2
import matplotlib.pyplot as plt
import numpy as np
import random


SIZE = 40
mask = np.zeros([SIZE, SIZE],dtype=np.uint8)
contours = np.array([[5,5],[5,10],[20,10],[20,5]])

cv2.fillPoly(mask, pts=[contours], color=(255))  # 内部为1
cv2.polylines(mask, pts=[contours], isClosed=True, color=64, thickness=3) # 边界为0.5

plt.imshow(mask)
plt.show()

# %%
center, radius = cv2.minEnclosingCircle(contours)
# %%
mask[int(center[1]),int(center[0])] = 255
plt.imshow(mask)
plt.show()
# %%
