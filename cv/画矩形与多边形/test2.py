#%%
import cv2
from cv2 import cv2
import matplotlib.pyplot as plt 
import numpy as np

mask = np.zeros([15,15])

for j in range(10):
    cv2.rectangle(mask, (5,5),(10,10),  (0.1*j+0.1), 10-j, 4)
plt.imshow(mask)
plt.show()

# for j in range(3):
#     cv2.rectangle(mask, (5,5),(10,10),  (0.3*j+0.1), 3-j, 4)
# plt.imshow(mask)
# plt.show()

# cv2.rectangle(mask, (5,5),(10,10),  (0.1), 3, 4)
# plt.imshow(mask)
# plt.show()

# cv2.rectangle(mask, (5,5),(10,10),  (0.5), 2, 4)
# plt.imshow(mask)
# plt.show()

# cv2.rectangle(mask, (5,5),(10,10),  (1), 1, 4)
# plt.imshow(mask)
# plt.show()
# %%

SIZE = 40
mask2 = np.zeros([SIZE, SIZE],dtype=np.float32)
contours = np.array([[1,1],[1,10],[20,10],[20,1]])

cv2.polylines(mask2, pts=[contours], isClosed=True, color=(0.5), thickness=5)
cv2.fillPoly(mask2, pts=[contours], color=(0.5))
center, radius = cv2.minEnclosingCircle(contours)
mask2[int(center[1]), int(center[0])] = 1.0

plt.imshow(mask2)
plt.show()
# %%
