# coding=utf-8
# 导入python包
import numpy as np
import cv2
# 读取图片并将其转化为灰度图片
image = cv2.imread('QQ.jpg')
orig = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# 利用cv2.minMaxLoc寻找到图像中最亮和最暗的点
(minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)
# 在图像中绘制结果
cv2.circle(image, maxLoc, 5, (255, 0, 0), 2)
# 应用高斯模糊进行预处理（由找点变成找区域）
gray = cv2.GaussianBlur(gray, (59,59), 0)
# 利用cv2.minMaxLoc寻找到图像中最亮和最暗的区域
(minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)
image1 = orig.copy()
cv2.circle(image1, maxLoc, 59, (255, 0, 0), 2)
# 显示结果
result = np.hstack([orig, image, image1])
cv2.imwrite("2.jpg", result)
cv2.imshow("Robust", result)
cv2.waitKey(0)

