#%%
import cv2
from cv2 import cv2
import matplotlib.pyplot as plt 
import numpy as np
import random




# %%
SIZE = 20

blank = np.zeros(shape=[SIZE, SIZE],dtype=np.uint8) 
for i in range(5):  
    cv2.circle(blank, (int(SIZE/2), int(SIZE/2)), i, (255-i*40), 1)
# plt.imshow(blank)
# plt.show()


#直接阈值化是对输入的单通道矩阵逐像素进行阈值分割。
ret, binary = cv2.threshold(blank, 200, 255, cv2.THRESH_BINARY)
print("threshold value %s"%ret)
# print(np.unicode(binary))
# plt.imshow(binary)
# plt.show()
contours, hier = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for c in contours:
    area = cv2.contourArea(c)  #  面积

    (x, y), radius = cv2.minEnclosingCircle(c)  #  外接圆
    Rectx, Recty, Rectw, Recth = cv2.boundingRect(c) # 矩形框
    imgTem =blank[Recty:Recty+Recth, Rectx:Rectx+Rectw]
    # 利用cv2.minMaxLoc寻找到图像中最亮和最暗的点
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(imgTem)
    print(maxLoc)
    print(imgTem)

    # center = (int(x), int(y))
    # img2 = cv2.circle(img2, center, radius+8, (0, 255, 0), 1) # 画圆
    # text = "radius:" + str(radius) + " " + "area:" + str(area)
    # img2 = cv2.putText(img2, text, center, cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2) #字符串

# %%
