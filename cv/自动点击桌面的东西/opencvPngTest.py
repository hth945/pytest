#%%
import os
import time
import datetime
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cv2
from cv2 import cv2
import pyautogui
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def myImgShow(img):
    plt.figure("Image") # 图像窗口名称
    plt.imshow(img)
    plt.axis('on') # 关掉坐标轴为 off
    plt.title('image') # 图像题目
    plt.show()

def getPintNow(template, region=None):
    img = pyautogui.screenshot(region=region)
    imgNP = np.asarray(img)
    result = cv2.matchTemplate(imgNP, template, cv2.TM_SQDIFF_NORMED)
    # result = cv2.matchTemplate(imgNP, template, cv2.TM_SQDIFF)
    # print(result.shape)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    imgNP = imgNP.copy()
    cv2.rectangle(imgNP, (min_loc[0], min_loc[1]),(min_loc[0] + template.shape[1], min_loc[1] + template.shape[0]),(255, 0, 0),4)
    # myImgShow(imgNP)
    px = min_loc[0] + template.shape[1] / 2
    py = min_loc[1] +  template.shape[0] / 2
    return imgNP, px,py,min_val# /(template.shape[0]*template.shape[1])


template = cv2.imread('2.jpg')
template = cv2.cvtColor(template, cv2.COLOR_BGR2RGB)

imgTem, px, py, min_val = getPintNow(template,region=(200,200, 300, 400))
# myImgShow(imgTem)
print(px,py,min_val)

imgTem, px, py, min_val = getPintNow(template,region=(0,0, 300, 400))
# myImgShow(imgTem)
print(px,py,min_val)

#%%
template = cv2.imread('2.jpg')
template = cv2.cvtColor(template, cv2.COLOR_BGR2RGB)
#%%
img = pyautogui.screenshot(region=(0,0, 300, 400))
imgNP = np.asarray(img)
result = cv2.matchTemplate(imgNP, template, cv2.TM_SQDIFF)
# 返回匹配的最小坐标
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
# a = cv2.minMaxLoc(result)[2]
# print(a)
cv2.rectangle(imgNP, (min_loc[0], min_loc[1]),(min_loc[0] + template.shape[1], min_loc[1] + template.shape[0]),(255, 0, 0),4)
myImgShow(imgNP)
px = min_loc[0] + template.shape[1] / 2
py = min_loc[1] +  template.shape[0] / 2
# %%




# %%
result.shape
# %%
myImgShow(template)
myImgShow(imgNP)
# %%
np.max(result)
# %%
np.min(result)
# %%
