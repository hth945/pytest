#%%
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
import os, sys, zipfile
import urllib.request
import cv2
import pylab
import os
import numpy as np
import random
from cv2 import cv2
import shutil
#%%
# Setup data paths
dataDir = '../../dataAndModel/data/coco'
dataType = 'val2017'
annDir = '{}/annotations'.format(dataDir)
annZipFile = '{}/annotations_train{}.zip'.format(dataDir, dataType)
annFile = '{}/instances_{}.json'.format(annDir, dataType)
annURL = 'http://images.cocodataset.org/annotations/annotations_train{}.zip'.format(dataType)
print (annDir)
print (annFile)
print (annZipFile)
print (annURL)
# Download data if not available locally
if not os.path.exists(annDir):
    os.makedirs(annDir)
if not os.path.exists(annFile):
    if not os.path.exists(annZipFile):
        print ("Downloading zipped annotations to " + annZipFile + " ...")
        with urllib.request.urlopen(annURL) as resp, open(annZipFile, 'wb') as out:
            shutil.copyfileobj(resp, out)
        print ("... done downloading.")
    print ("Unzipping " + annZipFile)
    with zipfile.ZipFile(annZipFile,"r") as zip_ref:
        zip_ref.extractall(dataDir)
    print ("... done unzipping")
print ("Will use annotations in " + annFile)
#%%

cocoRoot = "../../dataAndModel/data/coco/"
dataType = "val2017"

annFile = os.path.join(cocoRoot, f'annotations/instances_{dataType}.json')
print(f'Annotation file: {annFile}')
#%%
# # initialize COCO api for instance annotations
coco=COCO(annFile)

# 利用getCatIds函数获取某个类别对应的ID，
# 这个函数可以实现更复杂的功能，请参考官方文档
ids = coco.getCatIds('person')[0]
print(f'"person" 对应的序号: {ids}')
#%%
# 利用loadCats获取序号对应的文字类别
# 这个函数可以实现更复杂的功能，请参考官方文档
cats = coco.loadCats(1)
print(f'"1" 对应的类别名称: {cats}')

# 获取包含person的所有图片
imgIds = coco.getImgIds(catIds=[1])
print(f'包含person的图片共有：{len(imgIds)}张')

# 获取包含dog的所有图片
id = coco.getCatIds(['dog'])[0]
imgIds = coco.catToImgs[id]
print(f'包含dog的图片共有：{len(imgIds)}张, 分别是：')
print(imgIds)


#%%
imgId = imgIds[10]

imgInfo = coco.loadImgs(imgId)[0]
print(f'图像{imgId}的信息如下：\n{imgInfo}')

imPath = os.path.join(cocoRoot, 'images', dataType, imgInfo['file_name']) 
print(imPath) 
import skimage.io as io        
im = io.imread(imgInfo['coco_url']) 
# im = cv2.imread(imPath)
print(imgInfo['coco_url'])
plt.imshow(im)
plt.show()
#%%
# 获取该图像对应的anns的Id
annIds = coco.getAnnIds(imgIds=imgInfo['id'])
print(f'图像{imgInfo["id"]}包含{len(annIds)}个ann对象，分别是:\n{annIds}')
anns = coco.loadAnns(annIds)

coco.showAnns(anns)


print(f'ann{annIds[3]}对应的mask如下：')
mask = coco.annToMask(anns[3])
plt.imshow(mask)
plt.axis('off')
plt.show()
