#%%

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import cv2
import numpy as np
import shutil
import random
from zipfile import ZipFile 

rootPath = '..\..\dataAndModel\data\mnist\\'
for file in ["train", "test"]:
    path = rootPath + file 
    print(os.listdir(path))

# %%
image_paths = [rootPath + 'train\\' + file  for file in os.listdir(rootPath + 'train') ]
# %%
image_paths
# %%
image = cv2.imread(image_paths[0])
# %%
image.shape
# %%
image_paths[0]
# %%
