# -*- coding: utf-8 -*-
#%%
import json
with open('datalab/annotations.json') as f:
    a=json.load(f)
#%%
co=0
for c in a['annotations']:
    if c['image_id']==2162:
        co+=1
print(co)

print('标签类别:')
print('类别数量：',len(a['categories']))
print(a['categories'])

total=[]
for img in a['images']:
    hw=(img['height'],img['width'])
    total.append(hw)
unique=set(total)
for k in unique:
    print('长宽为(%d,%d)的图片数量为：'%k,total.count(k))
# "images": [{"file_name": "img_0017151.jpg", "height": 492, "id": 1, "width": 658},
#  "annotations": [{"area": 2522.739400000001, "iscrowd": 0, "image_id": 1, "bbox": [165.14, 53.71, 39.860000000000014, 63.29], "category_id": 2, "id": 213}, {"area": 207.50240000000025, "iscrowd": 0, "image_id": 2, "bbox": [465.71, 314.86, 13.580000000000041, 15.279999999999973], "category_id": 5, "id": 1169}
# "categories": [{"supercategory": "\u74f6\u76d6\u7834\u635f", "id": 1, "name": "\u74f6\u76d6\u7834\u635f"},
#%%
import pandas as pd
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.family']='sans-serif'
plt.rcParams['figure.figsize'] = (10.0, 10.0)

#%%
n = 0
for i in a['annotations']:
    if i['image_id'] == a['images'][4]['id']:
        n += 1
print(n)

for c in a['categories']:
    if c['id'] == 0:
        print(c['name'])
        
#%%
data=[]
annotas = a['annotations']
for img in a['images']:
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
cap=[]
for im in data:
    if im['height']== 492:
        cap.append(im)

# %%
len(cap)
q=[]
for c in cap:
    q.append(len(c['annotations']))
columns=set(q)
q_df=pd.DataFrame([q.count(cu) for cu in columns],index=columns,columns=['标注数量'])
q_df.plot(kind='bar')  

# %%
from PIL import Image, ImageFont, ImageDraw
import numpy as np
import colorsys
import matplotlib.pyplot as plt
def plot_imgs(img_data,gap=10,path=''):

    category_dic= {1: '瓶盖破损',
    9: '喷码正常',
    5: '瓶盖断点',
    3: '瓶盖坏边',
    4: '瓶盖打旋',
    0: '背景',
    2: '瓶盖变形',
    8: '标贴气泡',
    6: '标贴歪斜',
    10: '喷码异常',
    7: '标贴起皱'}

    files_name=img_data['file_name']
    img_annotations=img_data['annotations']
    n=len(img_annotations)
    boxs=np.zeros((n,4))
    tag=[]
    img=Image.open('./datalab/images/'+files_name) # 图片路径
    img_w=img.size[0]
    img_h=img.size[1]
    for i in range(n):
        bbox=img_annotations[i]['bbox']
        tag.append(category_dic[img_annotations[i]['category_id']])
        y1 = max(0, np.floor(bbox[1] + 0.5).astype('int32'))
        x1 = max(0, np.floor(bbox[0] + 0.5).astype('int32'))
        y2 = min(img_h, np.floor(bbox[1]+bbox[3] + 0.5).astype('int32'))
        x2= min(img_w, np.floor(bbox[0]+bbox[2] + 0.5).astype('int32'))
        boxs[i]=[x1,y1,x2,y2]

    
    font = ImageFont.truetype(font="simsun.ttc",size=np.floor(3.5e-2 * img_w).astype('int32'),encoding="unic")
    hsv_tuples = [(x / n, 1., 1.)for x in range(n)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                    colors))
    for index in range(len(boxs)):
        draw = ImageDraw.Draw(img)
        label_size = draw.textsize(tag[index], font)
        text_origin = np.array([20,25+index*label_size[1]])
        for i in range(gap):
            draw.rectangle(
            [boxs[index][0] + i, boxs[index][1] + i, boxs[index][2] - i, boxs[index][3] - i],outline=colors[index])
    #     draw.rectangle(list(),outline=colors[index])
        draw.rectangle( [tuple(text_origin), tuple(text_origin + label_size)],
                    fill=colors[index])
        draw.text(text_origin,tag[index], fill=(0, 0, 0), font=font)
    plt.imshow(img)

for c in cap:
    if len(c['annotations'])==21:
        print(c)
        break
plot_imgs(c,gap=2,path='') # 图片没上传上来 所以看不到

# %%


# %%
