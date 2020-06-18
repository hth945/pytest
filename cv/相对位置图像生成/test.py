#%%
import cv2
from cv2 import cv2
import matplotlib.pyplot as plt 
import numpy as np
import random


def makeResizeImage(image, ratio,points=None):
    if points != None:
        points = points * ratio
        image = cv2.resize(image, (int(image.shape[1] * ratio),  int(image.shape[0]*ratio)))
        return image, points
    else:
        image = cv2.resize(image, (int(image.shape[1] * ratio),  int(image.shape[0]*ratio)))
        return image


def makeSpinImage(image, angle,points=None):
    (h,w) = image.shape[:2]
    (cx,cy) = (w/2,h/2)
    M = cv2.getRotationMatrix2D((cx,cy),-angle,1.0)
    cos = np.abs(M[0,0])
    sin = np.abs(M[0,1])
    nW = int((h*sin)+(w*cos))
    nH = int((h*cos)+(w*sin))
    M[0,2] += (nW/2) - cx
    M[1,2] += (nH/2) - cy
    if points != None:
        points2 = np.zeros_like(points)
        for i in range(points.shape[0]):
            tem = np.array([[points[i,1]],[points[i,0]],[1]])
            tem2 = np.matmul(M, tem)
            points2[i,0] = tem2[1,0]
            points2[i,1] = tem2[0,0]
        return cv2.warpAffine(image,M,(nW,nH)),points2
    else:
        return cv2.warpAffine(image,M,(nW,nH))


SIZE = 20
blank = np.ones(shape=[SIZE, SIZE, 3])
blank[:,:,1:3] = 0
plt.imshow(blank)
plt.show()
img2 = makeSpinImage(blank, 45)
plt.imshow(img2)
plt.show()

# blank = np.zeros(shape=[SIZE, SIZE, 3]) 
# cv2.rectangle(blank, (5,5),(10,10),  (0.5,0,0), -1, 4)
# plt.imshow(blank)
# plt.show()

# %%
kx = 50,
ky = 50
kangle = 37

sx = 250
sy = 256
sangle = 70

simgSrc = cv2.imread("imgSrc.bmp")

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
kimgSrc = cv2.dilate(simgSrc, kernel)  # 膨胀
# kimgSrc = makeResizeImage(simgSrc,1.1)
kimgSrc[:,:,1:3] = 0
kimgSrc[:,:,0] = kimgSrc[:,:,0] / 255.0
simgSrc[:,:,0:2] = 0
simgSrc[:,:,2] = simgSrc[:,:,2] / 255.0

blank = np.zeros(shape=[224, 224, 3]) 
lab = np.array([112,112,37,112,112,50],dtype=np.int32)

tem = makeSpinImage(kimgSrc, lab[2])
startx = int(lab[0]-tem.shape[0]/2)
starty = int(lab[1]-tem.shape[1]/2)
blank[startx:startx+tem.shape[0], starty:starty+tem.shape[1], :] += tem

tem = makeSpinImage(simgSrc, lab[5])
startx = int(lab[3]-tem.shape[0]/2)
starty = int(lab[4]-tem.shape[1]/2)
blank[startx:startx+tem.shape[0], starty:starty+tem.shape[1], :] += tem
plt.imshow(blank)
plt.show()
# %%

# %%
kimgSrc.shape

# %%
blank.shape

# %%
blank

# %%
kimgSrc

# %%
