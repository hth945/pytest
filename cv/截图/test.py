#%%
import cv2
import imglable
import numpy as np
from iouTool import *

global point1, point2,flag
def on_mouse(event, x, y, flags, param):
    global img, point1, point2
    img, iou, ious = param
    img2 = img.copy()
    if event == cv2.EVENT_LBUTTONDOWN:         #左键点击
        point1 = (x,y)
        # cv2.circle(img2, point1, 10, (0,255,0), 1)  # 画圆
        # cv2.imshow('image', img2)
    elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):               #按住左键拖曳
        cv2.rectangle(img2, point1, (x,y), (255,0,0), 1)  # 画矩形
        cv2.imshow('image', img2)
    elif event == cv2.EVENT_LBUTTONUP:         #左键释放
        point2 = (x,y)
        cv2.rectangle(img2, point1, point2, (0,0,255), 1)
        cv2.imshow('image', img2)
        min_x = min(point1[0], point2[0])
        min_y = min(point1[1], point2[1])
        width = abs(point1[0] - point2[0])
        height = abs(point1[1] - point2[1])
        if width > 0 and height > 0:
            cut_img = img[min_y:min_y+height, min_x:min_x+width]
            cv2.imshow('test', cut_img)
            iou[0] = [min_x, min_y, width, height]  # x,y 宽 高
            drowIOU(img,ious,iou[0])
        # print(iou)

img = cv2.imread('1.png')
cv2.namedWindow('image')
cv2.imshow('image', img)
ious = []
iou = [[-1]]
parm = [img, iou, ious]
cv2.setMouseCallback('image', on_mouse,parm)
while (True):
    key = (cv2.waitKey(0) & 0xFF)
    if key == ord('s') and iou[0][0] != -1:   # 保存
        ious.append(iou[0])
        iou[0] = [-1]
        print(ious)
        drowIOU(img, ious)
    elif key == ord('d') and len(ious) != 0:  # 删除
        ious.pop()
        iou[0] = [-1]
        print(ious)
        drowIOU(img, ious)
    elif key == ord('q'):   # 推出并保存

        break



# # 标记图片
# def main(filePath,pklPath):
#     iou = [[-1]]
#     # filePath = 'yang/A3.png'
#     imgL = imglable.ImageLable()
#
#     pix = [70,2048,0,2048]
#     imgL.load(pklPath)
#     print(imgL.lables)
#     ious = []
#     if filePath in imgL.paths:
#         index = imgL.paths.index(filePath)
#         print(index)
#         ious = imgL.lables[index].copy()
#     print(len(ious))
#     # ious = []
#
#     img = cv2.imread(filePath)
#     img = img[pix[0]:pix[1], pix[2]:pix[3]]
#     # cv2.namedWindow('image',cv2.WINDOW_NORMAL)
#     cv2.namedWindow('image')
#     cv2.imshow('image', img)
#     parm = [img, iou, ious]
#     cv2.setMouseCallback('image', on_mouse,parm)
#     while (True):
#         key = (cv2.waitKey(0) & 0xFF)
#         if key == ord('s') and iou[0][0] != -1:   # 保存
#             ious.append(iou[0])
#             iou[0] = [-1]
#             print(ious)
#             drowIOU(img, ious)
#         elif key == ord('d') and len(ious) != 0:  # 删除
#             ious.pop()
#             iou[0] = [-1]
#             print(ious)
#             drowIOU(img, ious)
#         elif key == ord('q'):   # 推出并保存
#             imgL.append(filePath, pix, ious)
#             imgL.save(pklPath)
#             break
