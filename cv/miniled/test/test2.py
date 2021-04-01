#%%

import os
import cv2
from cv2 import cv2
import matplotlib.pyplot as plt 
import numpy as np

mask = np.zeros([24, 32],dtype=np.uint8)
contours = np.array([[1,1],[1,10],[20,10],[20,1]])

cv2.fillPoly(mask, pts=[contours], color=(255))  # 内部为1
cv2.polylines(mask, pts=[contours], isClosed=True, color=64, thickness=3) # 边界为0.5

N = 15
plt.figure(figsize=(N, N))
plt.imshow(mask, cmap='gray')
plt.show()


outImg = np.zeros([6, 128],dtype=np.uint8)

for i in range(4):
    outImg[0:6,32*i:32*(i+1)]=mask[6*i:6*(i+1),:]

plt.figure(figsize=(N, N))
plt.imshow(outImg, cmap='gray')
plt.show()
#%%

def fileToC(image, out_file):
    OUT_FILE=out_file
    x,y = image.shape

    out_file = open(OUT_FILE+'.c', 'w')
    array_name = os.path.basename(OUT_FILE)
    out_file.write('const unsigned char %s[%d][%d] = {    '%(array_name, x,y))

    for h in image:
        out_file.write('\n    ')
        for d in h:
            out_file.write('0x%02x'%d+', ')

    out_file.write('\n};')
    out_file.close()
    print('complete')
fileToC(outImg,'htmlData')


#%%

# %%
