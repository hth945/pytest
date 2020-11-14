import os
import time
import datetime
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cv2
import numpy as np
import pyautogui
import random

def get_screenshot(id=0):
    os.system('adb shell screencap -p /sdcard/%s.png' % str(0))
    os.system('adb pull /sdcard/0.png ./%s.png ' % str(id))
    img = cv2.imread('./%s.png'  % str(id))
    return img

def adbClick(x,y):
    cmd = ('adb shell input swipe %i %i %i %i %i' ) % (x + random.randint(-10, 10), y + random.randint(-10, 10),
                                                     x + random.randint(-10, 10), y + random.randint(-10, 10),
                                                     200+ random.randint(-100, 100)
                                                     )
    os.system(cmd)
    print(cmd)

def myImgShow(img):
    plt.figure("Image") # 图像窗口名称
    plt.imshow(img)
    plt.axis('on') # 关掉坐标轴为 off
    plt.title('image') # 图像题目
    plt.show()

def getPintNow(imgNP, template, region=None):
    result = cv2.matchTemplate(imgNP, template, cv2.TM_SQDIFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    imgNP = imgNP.copy()
    cv2.rectangle(imgNP, (min_loc[0], min_loc[1]),(min_loc[0] + template.shape[1], min_loc[1] + template.shape[0]),(255, 0, 0),4)
    # myImgShow(imgNP)
    px = min_loc[0] + template.shape[1] / 2
    py = min_loc[1] +  template.shape[0] / 2
    return imgNP, px,py,min_val


def runClick(templatePath,minRange=0.02,lastTemplatePath=None,lastTemplateMinRange=0.02,maxTime=2000):
    i = 0
    begin_time = time.time()
    template = cv2.imread(templatePath)
    lastTemplate = None
    if lastTemplatePath!= None:
        lastTemplate = cv2.imread(lastTemplatePath)
    while True:

        img = get_screenshot()
        imgTem, px, py, min_val = getPintNow(img, template)
        # cv2.imshow('imgTem', imgTem)
        # cv2.waitKey(1)
        # cv2.imwrite('imgTem.png', imgTem)
        # myImgShow(imgTem)
        print(px, py, min_val)
        if (min_val < minRange):
            print('click')
            adbClick(px, py)
            return 0
        if lastTemplatePath != None:
            imgTem, px, py, min_val = getPintNow(img, lastTemplate)
            print('last :', px, py, min_val)
            if (min_val < lastTemplateMinRange):
                print('click')
                adbClick(px, py)
        time.sleep(1)
        # if time.time() - begin_time > (maxTime/1000.0):
        #     raise Exception("templatePath :", templatePath)

i = 0
while i < 100:
    i+=1
    print('1')
    runClick('startXL.png',lastTemplatePath='endX.png', lastTemplateMinRange=0.05)
    print('2')
    runClick('startX2.png')
    print('3')
    runClick('endX.png',minRange=0.05)

# img = get_screenshot()
# cv2.imshow('imgTem', img)
cv2.waitKey(0)
