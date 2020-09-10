#%%
import cv2
from cv2 import cv2
import numpy as np
import shutil
import os
import random

    
def compute_iou(box1, box2):
    # xmin, ymin, xmax, ymax
    A1 = (box1[2] - box1[0])*(box1[3] - box1[1])
    A2 = (box2[2] - box2[0])*(box2[3] - box2[1])

    xmin = max(box1[0], box2[0])
    ymin = max(box1[1], box2[1])
    xmax = min(box1[2], box2[2])
    ymax = min(box1[3], box2[3])

    if ymin >= ymax or xmin >= xmax: return 0
    return  ((xmax-xmin) * (ymax - ymin)) / (A1 + A2)


def make_image(data, image_path, ratio=1):
    blank = data[0]
    boxes = data[1]
    label = data[2]

    ID = image_path.split("/")[-1][0]
    image = cv2.imread(image_path)
    image = cv2.resize(image, (int(28*ratio), int(28*ratio)))
    h, w, c = image.shape

    while True:
        xmin = np.random.randint(0, SIZE-w, 1)[0]
        ymin = np.random.randint(0, SIZE-h, 1)[0]
        xmax = xmin + w
        ymax = ymin + h
        box = [xmin, ymin, xmax, ymax]

        iou = [compute_iou(box, b) for b in boxes]
        if max(iou) < 0.02:
            boxes.append(box)
            label.append(ID)
            break

    for i in range(w):
        for j in range(h):
            x = xmin + i
            y = ymin + j
            blank[y][x] = image[j][i]

    # cv2.rectangle(blank, (xmin, ymin), (xmax, ymax), [0, 0, 255], 2)
    return blank

rootPath = '..\..\dataAndModel\data\mnist\\'
image_paths = [rootPath + 'train\\' + file  for file in os.listdir(rootPath + 'train') ]

SIZE =416
image_sizes = [3, 6, 3]

blanks = np.ones(shape=[SIZE, SIZE, 3]) * 255
bboxes = [[0,0,1,1]]
labels = [0]
data = [blanks, bboxes, labels]
bboxes_num = 0

# ratios small, medium, big objects
ratios = [[0.5, 0.8], [1., 1.5, 2.], [3., 4.]]
for i in range(len(ratios)):
    N = random.randint(0, image_sizes[i])
    if N !=0: bboxes_num += 1
    for _ in range(N):
        ratio = random.choice(ratios[i])
        idx = random.randint(0, len(image_paths)-1)
        data[0] = make_image(data, image_paths[idx], ratio)

images_path = './mnist/'
image_num = 0
image_path = os.path.realpath(os.path.join(images_path, "%06d.jpg" %(image_num+1)))
cv2.imwrite(image_path, data[0])      
# %%
