#%%
import json
import cv2
from cv2 import cv2
import numpy as np

with open('datalab/annotations.json') as f:
    a=json.load(f)

data=[]
annotas = a['annotations']
for img in a['images']:
    if img['height'] != 492:
        continue
    sample_img = img
    sample_annota_list = []
    for per in annotas:
        if img['id'] == per['image_id'] and per['category_id'] != 0:
            sample_annota_list.append(per)
    for k in sample_annota_list:
        annotas.remove(k)
    sample_img['annotations'] = sample_annota_list
    data.append(sample_img)

# %%
def imgShow(imgsinfo, img_data):
    img = cv2.imread('./datalab/images/'+img_data['file_name'])
    imginfo = {}
    imginfo['img'] = img
    img_annotations=img_data['annotations']
    n = len(img_annotations)
    boxs=np.zeros((n,5),dtype=np.int32)
    for i in range(n):
        bbox = img_annotations[i]['bbox']
        boxs[i,0] = img_annotations[i]['category_id']
        y1 = max(0, np.floor(bbox[1] + 0.5).astype('int32'))
        x1 = max(0, np.floor(bbox[0] + 0.5).astype('int32'))
        y2 = min(img.shape[0], np.floor(bbox[1]+bbox[3] + 0.5).astype('int32'))
        x2= min(img.shape[1], np.floor(bbox[0]+bbox[2] + 0.5).astype('int32'))
        boxs[i,1:5]=[x1,y1,x2,y2]
        
    imginfo['boxs'] = boxs
    imgsinfo.append(imginfo)

imgsinfo = [] 
for d in data:
    imgShow(imgsinfo,d)
    break
print(len(imgsinfo))
print(imgsinfo)
# %%
import  os
print(os.getcwd()) #获取当前工作目录路径

# %%
