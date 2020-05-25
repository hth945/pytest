#%%
import cv2
from cv2 import cv2
import matplotlib.pyplot as plt
import numpy as np
import random

def makeResizeImage(image, ratio,points):
    points = points * ratio
    image = cv2.resize(image, (int(image.shape[1] * ratio),  int(image.shape[0]*ratio)))
    return image, points

def getPointsImg(image,points):
    img = np.zeros_like(image)
    for i in range(points.shape[0]):
        img[int(points[i,0]), int(points[i,1])] = 255
    return img

def drowPointsLableImg(image,pointsLable):
    for points in pointsLable:
        for i in range(points.shape[0]):
            image[int(points[i,0]), int(points[i,1])] = 255

def makeSpinImage(image, angle,points):
    (h,w) = image.shape[:2]
    (cx,cy) = (w/2,h/2)
    M = cv2.getRotationMatrix2D((cx,cy),-angle,1.0)
    cos = np.abs(M[0,0])
    sin = np.abs(M[0,1])
    nW = int((h*sin)+(w*cos))
    nH = int((h*cos)+(w*sin))
    M[0,2] += (nW/2) - cx
    M[1,2] += (nH/2) - cy
    points2 = np.zeros_like(points)
    for i in range(points.shape[0]):
        tem = np.array([[points[i,1]],[points[i,0]],[1]])
        tem2 = np.matmul(M, tem)
        points2[i,0] = tem2[1,0]
        points2[i,1] = tem2[0,0]
    return cv2.warpAffine(image,M,(nW,nH)),points2

def compute_iou(box1, box2):
    """xmin, ymin, xmax, ymax"""
    A1 = (box1[2] - box1[0])*(box1[3] - box1[1])
    A2 = (box2[2] - box2[0])*(box2[3] - box2[1])
    xmin = max(box1[0], box2[0])
    ymin = max(box1[1], box2[1])
    xmax = min(box1[2], box2[2])
    ymax = min(box1[3], box2[3])
    if ymin >= ymax or xmin >= xmax: return 0
    return  ((xmax-xmin) * (ymax - ymin)) / (A1 + A2)

# # %%
# SIZE = 256
# imgSrc = cv2.imread("imgSrc.bmp", cv2.IMREAD_GRAYSCALE)
# pointsSrc = np.array([[1,6],[1,22],[9,6],[9,22]])
#
# blank = np.zeros(shape=[SIZE, SIZE])
# boxes = []
# pointLabels = []
# for i in range(5):
#     angle = random.randint(0, 360)
#     img, points = makeSpinImage(imgSrc, angle, pointsSrc)
#     xmin = np.random.randint(0, SIZE-img.shape[0], 1)[0]
#     ymin = np.random.randint(0, SIZE-img.shape[1], 1)[0]
#     xmax = xmin + img.shape[0]
#     ymax = ymin + img.shape[1]
#     box = [xmin, ymin, xmax, ymax]
#     if len(boxes) > 0:
#         iou = [compute_iou(box, b) for b in boxes]
#         if max(iou) > 0.02:
#             continue
#     points += np.array([xmin, ymin])
#     boxes.append(box)
#     pointLabels.append(points)
#     blank[xmin:xmax, ymin:ymax] = img

# %%
