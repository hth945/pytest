#%%
# 0	建筑
# 1	耕地
# 2	林地
# 3	水体
# 4	道路
# 5	草地
# 6	其他
# 255	未标注区域 (255,255,255)
import cv2
from cv2 import cv2
import matplotlib.pyplot as plt 
import numpy as np
from PIL import Image

LabName ={0:'建筑', 1:'耕地', 2:'林地', 3:'水体', 4:'道路', 5:'草地', 6:'其他', 255:'未标注区域'}

path = 'D:/sysDef/Documents/GitHub/pytest/dataAndModel/data/bcdi/'
data='img_train/T000000.jpg lab_train/T000000.png'
imgPath,labPath=data.split(' ') 
imgPath =path + imgPath
labPath =path + labPath

def showLabimg(labPath):
    lab=cv2.imread(labPath)
    # plt.imshow(lab)
    # plt.show()
    img = Image.open(labPath)
    np_img = np.array(img)
    labels = list(set(np_img.flatten()))
    print("np_img:", np_img)
    print("label:", labels)

    LabName ={0:'建筑', 1:'耕地', 2:'林地', 3:'水体', 4:'道路', 5:'草地', 6:'其他', 255:'未标注区域'}
    for lab in labels:
        s = np.zeros((256,256))
        s[np_img == lab]=1
        # plt.imshow(s)
        # plt.show()
        print(lab)
        print(LabName[lab])
    return np_img
showLabimg(path+'lab_train/T000001.png')
# img = showLabimg('./T000054.png')
# showLabimg('./T000067.png')

showLabimg('D:\sysDef\download\A145981.png')
# %%
