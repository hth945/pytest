import cv2
import imglable
import numpy as np


def drowIOU(img,ious,iou=None,name=None):
    img = img.copy()
    if iou != None:
        cv2.rectangle(img, (iou[0], iou[1]), (iou[0] + iou[2], iou[1] + iou[3]), (0, 0, 255),1)  # 画矩形

    for iou in ious:
        cv2.rectangle(img, (iou[0], iou[1]), (iou[0] + iou[2], iou[1] + iou[3]), (0, 255, 0), 1)  # 画矩形
    if name is None:
        name = 'imgSave'
    cv2.imshow(name, img)


# iou 目标区域 pix 图片中哪片范围内的像素
def getContainIou(iou, pix):
    # min_x, min_y, width, height = iou
    # x1,x2,y1,y2 = pix # [60,2048,0,2048]
    # print(iou,pix)
    x1 = max(iou[0]-128+10, pix[0])  # 超过0
    x1 = min(x1, pix[1]-128)
    x2 = min(128+iou[2]-20,pix[1]-x1-128)

    y1 = max(iou[1] - 128 + 10, pix[2])  # 超过0
    y1 = min(y1, pix[3] - 128)
    y2 = min(128 + iou[3] - 20, pix[3] - y1)

    return x1, x2, y1, y2


## 检查ious中是否有重合部分
def getCheckIou(iouC, ious):
    min_x, min_y, width, height = iouC
    b1_x0, b1_y0, b1_x1, b1_y1 = min_x, min_y, min_x+width, min_y+height
    for iou in ious:
        min_x, min_y, width, height = iou
        b2_x0, b2_y0, b2_x1, b2_y1 = min_x, min_y, min_x+width, min_y+height
        int_x0 = max(b1_x0, b2_x0)
        int_y0 = max(b1_y0, b2_y0)
        int_x1 = min(b1_x1, b2_x1)
        int_y1 = min(b1_y1, b2_y1)
        int_area = (int_x1 - int_x0) * (int_y1 - int_y0)
        # print(b2_x0, b2_y0, b2_x1, b2_y1,'    ',b1_x0, b1_y0, b1_x1, b1_y1)
        # print(int_x1,int_x0,int_y1,int_y0)
        # print(int_area)
        if int_area > 4 and (int_x1 - int_x0) > 0:
            return 1
    return 0

# ## 检查ious中是否有重合部分
# def getCheckIou(iouC, ious):
#     min_x, min_y, width, height = iouC
#     b1_x0, b1_y0, b1_x1, b1_y1 = min_x+width, min_y+height, width, height
#     for iou in ious:
#         min_x, min_y, width, height = iou
#         b2_x0, b2_y0, b2_x1, b2_y1 = min_x + width, min_y + height, width, height
#         int_x0 = max(b1_x0, b2_x0)
#         int_y0 = max(b1_y0, b2_y0)
#         int_x1 = min(b1_x1, b2_x1)
#         int_y1 = min(b1_y1, b2_y1)
#         int_area = (int_x1 - int_x0) * (int_y1 - int_y0)
#         if int_area > 4:
#             return True
#     return False