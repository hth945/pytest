#%%

import torch
import torchvision
import argparse
import cv2
import numpy as np
import sys
import random

names = {'0': 'background', '1': 'person', '2': 'bicycle', '3': 'car', '4': 'motorcycle', '5': 'airplane', '6': 'bus', '7': 'train', '8': 'truck', '9': 'boat', '10': 'traffic light', '11': 'fire hydrant', '13': 'stop sign', '14': 'parking meter', '15': 'bench', '16': 'bird', '17': 'cat', '18': 'dog', '19': 'horse', '20': 'sheep', '21': 'cow', '22': 'elephant', '23': 'bear', '24': 'zebra', '25': 'giraffe', '27': 'backpack', '28': 'umbrella', '31': 'handbag', '32': 'tie', '33': 'suitcase', '34': 'frisbee', '35': 'skis', '36': 'snowboard', '37': 'sports ball', '38': 'kite', '39': 'baseball bat', '40': 'baseball glove', '41': 'skateboard', '42': 'surfboard', '43': 'tennis racket', '44': 'bottle', '46': 'wine glass', '47': 'cup', '48': 'fork', '49': 'knife', '50': 'spoon', '51': 'bowl', '52': 'banana', '53': 'apple', '54': 'sandwich', '55': 'orange', '56': 'broccoli', '57': 'carrot', '58': 'hot dog', '59': 'pizza', '60': 'donut', '61': 'cake', '62': 'chair', '63': 'couch', '64': 'potted plant', '65': 'bed', '67': 'dining table', '70': 'toilet', '72': 'tv', '73': 'laptop', '74': 'mouse', '75': 'remote', '76': 'keyboard', '77': 'cell phone', '78': 'microwave', '79': 'oven', '80': 'toaster', '81': 'sink', '82': 'refrigerator', '84': 'book', '85': 'clock', '86': 'vase', '87': 'scissors', '88': 'teddybear', '89': 'hair drier', '90': 'toothbrush'}

num_classes = 91
model = torchvision.models.detection.__dict__['fasterrcnn_resnet50_fpn'](num_classes=num_classes, pretrained=False)  
model = model.cuda()

save = torch.load('model_13.pth')
model.load_state_dict(save['model'])


src_img = cv2.imread('imgs/1.jpg')
img = cv2.cvtColor(src_img,cv2.COLOR_BGR2RGB)
img_tensor = torch.from_numpy(img/255.).permute(2,0,1).float().cuda()
input = []
input.append(img_tensor)
out = model(input)
boxes = out[0]['boxes']
labels = out[0]['labels']
scores = out[0]['scores']
for idx in range(boxes.shape[0]):
    if scores[idx] >= 0.8:
        x1, y1, x2, y2 = boxes[idx][0], boxes[idx][1], boxes[idx][2], boxes[idx][3]
        name = names.get(str(labels[idx].item()))
        x1 = int(x1)
        x2 = int(x2)
        y1 = int(y1)
        y2 = int(y2)
        print(x1,y1,x2,y2)
        cv2.rectangle(src_img,(x1,y1),(x2,y2),random_color(),thickness=2)
        cv2.putText(src_img, text=name, org=(x1, y1+10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
            fontScale=0.5, thickness=1, lineType=cv2.LINE_AA, color=(0, 0, 255))

cv2.imshow('result',src_img)
# %%
