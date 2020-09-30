#%%

import numpy as np
import glob
import random

def load_data(path):
    data = []
    with open(path) as f:
        while True:
            line = f.readline().split()
            if len(line) == 0:
                break
            for i, bbox in enumerate(line[1:]):
                bbox = bbox.split(",")
                xmin = np.float64(bbox[0])
                ymin = np.float64(bbox[1])
                xmax = np.float64(bbox[2])
                ymax = np.float64(bbox[3])
            # 得到宽高

            data.append([xmax-xmin,ymax-ymin])
    return np.array(data)


path = r'D:\sysDef\Documents\GitHub\pytest\dataAndModel\data\mnist\objtrainlab.txt'
    
# 载入所有的xml
# 存储格式为转化为比例后的width,height
data = load_data(path)

import matplotlib.pyplot as plt

#用绘图框架的plot()方法绘图, 样式为".", 颜色为红色
plt.plot(data[:,0], data[:,1],".", color = "r")

plt.show()
# %%
data.shape
# %%
archors = np.array([29,29, 56,56, 84,84, 112,112, 196,196, 252,252]).reshape(-1,2)
# %%
archors
# %%
plt.plot(data[:,0], data[:,1],".", color = "r")
plt.plot(archors[:,0], archors[:,1],".", color = "b")
plt.show()