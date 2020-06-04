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
