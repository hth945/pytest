#%%
import cv2
from cv2 import cv2
import numpy as np
import shutil
import os
import random
from PIL import Image
import matplotlib.pyplot as plt 

rootPath = '..\..\dataAndModel\data\mnist\\'

ID = random.randint(0, 10)
label_txt = rootPath + 'objtrainlab.txt'
image_info = open(label_txt).readlines()[ID].split()
print(image_info)
image_path = image_info[0]
image = cv2.imread(image_path)
for bbox in image_info[1:]:
    bbox = bbox.split(",")
    image = cv2.rectangle(image,(int(float(bbox[0])),
                                 int(float(bbox[1]))),
                                (int(float(bbox[2])),
                                 int(float(bbox[3]))), (255,0,0), 2)

plt.imshow(image)
plt.show()
# %%
image
# %%
