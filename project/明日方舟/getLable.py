import cv2
import numpy as np

def drowIOU(img,ious=None,iou=None,name=None):
    img = img.copy()
    if iou != None:
        cv2.rectangle(img, (iou[0], iou[1]), (iou[0] + iou[2], iou[1] + iou[3]), (0, 0, 255), 1)  # 画矩形
    if ious != None:
        for iou in ious:
            cv2.rectangle(img, (iou[0], iou[1]), (iou[0] + iou[2], iou[1] + iou[3]), (0, 255, 0), 1)  # 画矩形
    if name is None:
        name = 'imgSave'
    cv2.imshow(name, img)

global point1, point2,flag
def on_mouse(event, x, y, flags,param):
    global img, point1, point2
    img = param
    img2 = img.copy()
    if event == cv2.EVENT_LBUTTONDOWN:         #左键点击
        point1 = (x, y)
        # cv2.circle(img2, point1, 10, (0,255,0), 1)  # 画圆
        # cv2.imshow('image', img2)
    elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):               #按住左键拖曳
        cv2.rectangle(img2, point1, (x, y), (255, 0, 0), 1)  # 画矩形
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
        #     iou= [min_x, min_y, width, height]  # x,y 宽 高
        #     drowIOU(img,iou=iou)
        # print(iou)

img = cv2.imread('start1.png')
cv2.namedWindow('image')
cv2.imshow('image', img)
cv2.setMouseCallback('image', on_mouse,img)
while (True):
    key = (cv2.waitKey(0) & 0xFF)
    if key == ord('q'):   # 推出并保存

        break


