import cv2
from config import cfg
import numpy as np
import tensorflow as tf
import datetime
import os
import xml.etree.ElementTree as ET
import csv

# img = cv2.imread('../newLable/Image_20200416200723024.bmp')

modelTest = tf.keras.models.load_model('testmodel.h5')
modelTest.train = False
@tf.function
def runModel(imgTem):
    imgTem = tf.cast(imgTem, dtype=tf.float32)
    return modelTest(imgTem)


path = 'id'
if not os.path.exists(path):
    os.makedirs(path)

with open(path + '/id_'+datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S') + '.csv', 'w', newline='') as csvfile:
    fieldnames = ['picNumber', 'pointNumber', 'x', 'y', 'radius', 'area']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()  # 注意有写header操作
    data_info = []
    picNumber = 0

    for ann_dir in cfg.TEST.ANNOT_PATH:
        for anno in os.listdir(ann_dir):
            tree = ET.parse(os.path.join(ann_dir, anno))
            for elem in tree.iter():
                if 'path' == elem.tag:
                    imgP = ann_dir + '/../' + elem.text[elem.text.rfind('\\') + 1:]
                    img = cv2.imread(imgP)
                    # img = img[:3648,:3648,:]
                    # print(img.shape)
                    image_data = img[np.newaxis]
                    imgIn = image_data / 255.0
                    label = runModel(imgIn)

                    dic = dict()
                    dic['picNumber'] = picNumber
                    mask = label.numpy()[:,:,:,0]
                    # print(np.unique(mask))
                    # # print(image_data[0].shape)
                    # # print(np.unique(image_data[0]))
                    # cv2.imshow('1', image_data[0]/255.0)
                    # # cv2.imshow('2', img[0].numpy() / 255.0)
                    # cv2.imshow('mask', mask[0] * 1.0)

                    gray = np.uint8(mask[0] * 255.0)

                    ret, binary = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY)
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                    dst = cv2.dilate(binary, kernel)  # 膨胀
                    # cv2.imshow("dst1", dst)
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
                    dst = cv2.erode(dst, kernel)  # 腐蚀
                    # cv2.imshow("dst2", dst)

                    contours, hier = cv2.findContours(dst, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    img2 = np.uint8(image_data[0])
                    pointNumber = 0
                    for c in contours:
                        area = cv2.contourArea(c)  #  面积
                        if area > 3:
                            (x, y), radius = cv2.minEnclosingCircle(c)  #  外接圆
                            Rectx, Recty, Rectw, Recth = cv2.boundingRect(c)

                            dic['pointNumber'] = pointNumber
                            pointNumber += 1
                            dic['x'] = x
                            dic['y'] = y
                            dic['radius'] = radius
                            dic['area'] = area

                            writer.writerow(dic)
                            # 规范化为整数
                            center = (int(x), int(y))
                            radius = int(radius)
                            # 勾画圆形区域
                            img2 = cv2.circle(img2, center, radius+8, (0, 255, 0), 1)
                            text = "radius:" + str(radius) + " " + "area:" + str(area)
                            img2 = cv2.putText(img2, text, center, cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)

                    #  画出边界
                    #  cv2.drawContours(img2, contours, -1, (255, 255, 255), 1)

                    # cv2.imshow('img', img2)
                    print(picNumber)
                    cv2.imwrite(path + '/id_'+str(picNumber)+'_mask.png', gray)
                    cv2.imwrite(path + '/id_'+str(picNumber)+'_src.png', image_data[0])
                    picNumber += 1

                    # cv2.waitKey(0)
                    # if picNumber > 3:
                    #     exit()


