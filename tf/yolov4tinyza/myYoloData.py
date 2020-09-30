#%%
# 
from tqdm import tqdm
import time
import os
import numpy as np
import cv2
from PIL import Image
from utils import get_random_data, get_random_data_with_Mosaic
from matplotlib import pyplot as plt

def rand(a=0.0, b=1.0):
    return np.random.rand()*(b-a) + a

#---------------------------------------------------#
#   获得类和先验框
#---------------------------------------------------#
def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)


#---------------------------------------------------#
#   训练数据生成器
#---------------------------------------------------#
def data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes, mosaic=False):
    '''data generator for fit_generator'''
    n = len(annotation_lines)
    i = 0
    flag = True
    while True:
        image_data = []
        box_data = []
        for b in range(batch_size):
            if i==0:
                np.random.shuffle(annotation_lines)
            if mosaic:
                if flag and (i+4) < n:
                    image, box = get_random_data_with_Mosaic(annotation_lines[i:i+4], input_shape)
                    i = (i+1) % n
                else:
                    image, box = get_random_data(annotation_lines[i], input_shape)
                    i = (i+1) % n
                flag = bool(1-flag)
            else:
                image, box = get_random_data(annotation_lines[i], input_shape)
                i = (i+1) % n
            image_data.append(image)
            box_data.append(box)
        
        
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
        yield image_data, y_true[0], y_true[1]


#---------------------------------------------------#
#   读入xml文件，并输出y_true
#---------------------------------------------------#
def preprocess_true_boxes(true_boxes, input_shape, anchors, num_classes):
    assert (true_boxes[..., 4]<num_classes).all(), 'class id must be less than num_classes'
    # 一共有三个特征层数
    num_layers = len(anchors)//3
    # 先验框
    # 678为 142,110,  192,243,  459,401
    # 345为 36,75,  76,55,  72,146
    # 012为 12,16,  19,36,  40,28
    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers==3 else [[3,4,5], [1,2,3]]

    true_boxes = np.array(true_boxes, dtype='float32')
    input_shape = np.array(input_shape, dtype='int32') # 416,416
    # 读出xy轴，读出长宽
    # 中心点(m,n,2)
    boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2
    boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]
    # 计算比例
    true_boxes[..., 0:2] = boxes_xy/input_shape[:]
    true_boxes[..., 2:4] = boxes_wh/input_shape[:]

    # m张图
    m = true_boxes.shape[0]
    # 得到网格的shape为13,13;26,26;52,52
    grid_shapes = [input_shape//{0:32, 1:16, 2:8}[l] for l in range(num_layers)]
    # y_true的格式为(m,13,13,3,85)(m,26,26,3,85)(m,52,52,3,85)
    y_true = [np.zeros((m,grid_shapes[l][0],grid_shapes[l][1],len(anchor_mask[l]),5+num_classes),
        dtype='float32') for l in range(num_layers)]
    # [1,9,2]
    anchors = np.expand_dims(anchors, 0)
    anchor_maxes = anchors / 2.
    anchor_mins = -anchor_maxes
    # 长宽要大于0才有效
    valid_mask = boxes_wh[..., 0]>0

    print(true_boxes)

    for b in range(m):
        # 对每一张图进行处理
        wh = boxes_wh[b, valid_mask[b]]
        if len(wh)==0: continue
        # [n,1,2]
        wh = np.expand_dims(wh, -2)
        box_maxes = wh / 2.
        box_mins = -box_maxes

        # 计算真实框和哪个先验框最契合
        intersect_mins = np.maximum(box_mins, anchor_mins)
        intersect_maxes = np.minimum(box_maxes, anchor_maxes)
        intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        box_area = wh[..., 0] * wh[..., 1]
        anchor_area = anchors[..., 0] * anchors[..., 1]
        iou = intersect_area / (box_area + anchor_area - intersect_area)
        # 维度是(n) 感谢 消尽不死鸟 的提醒
        best_anchor = np.argmax(iou, axis=-1)

        for t, n in enumerate(best_anchor):
            for l in range(num_layers):
                if n in anchor_mask[l]:
                    # floor用于向下取整
                    i = np.floor(true_boxes[b,t,0]*grid_shapes[l][1]).astype('int32')
                    j = np.floor(true_boxes[b,t,1]*grid_shapes[l][0]).astype('int32')
                    # 找到真实框在特征层l中第b副图像对应的位置
                    k = anchor_mask[l].index(n)
                    c = true_boxes[b,t, 4].astype('int32')
                    # print(t,n,l)
                    # print(true_boxes[b,t, :])

                    y_true[l][b, j, i, k, 0:4] = true_boxes[b,t, 0:4]
                    y_true[l][b, j, i, k, 4] = 1
                    y_true[l][b, j, i, k, 5+c] = 1
                # else:
                #     print(t,n,l,'NG')


    return y_true

if __name__ == '__main__':
    annotation_path = r'D:\sysDef\Documents\GitHub\pytest\dataAndModel\data\mnist\objtrainlab.txt'
    anchors_path = 'yolo_anchors.txt'
    with open(annotation_path) as f:
        lines = f.readlines()
    import myYoloData
    anchors = myYoloData.get_anchors(anchors_path)

    line = lines[0].split()
    image = Image.open(line[0])
    iw, ih = image.size
    h, w = (419,419)
    box = np.array([[np.array(list(map(int, box.split(',')))) for box in line[1:]]])
    y_true = preprocess_true_boxes(box, (416,416), anchors, 10)
    y_true0 = y_true[0]
    y_true1 = y_true[1]

    image_data =  np.array(image,np.float32)/255 # cv2.cvtColor(, cv2.COLOR_RGB2HSV)
    # print(y_true)
    print(image_data.shape)
    print(y_true0.shape)
    print(y_true1.shape)
    

    
    print((np.reshape(y_true0,[-1,3,15])[:,:,4] > 0.5).sum())
    print((np.reshape(y_true1,[-1,3,15])[:,:,4] > 0.5).sum())

    import cv2
    from cv2 import cv2
    image = image_data.copy()


    lab = np.reshape(y_true0,[-1,3,15])
    bboxs = lab[lab[:,:,4] > 0.5][:,:4]
    for i in range(bboxs.shape[0]):
        bbox = bboxs[i]
        image = cv2.rectangle(image, (int(float(bbox[0]*416 - bbox[2]*416/2)),
                                    int(float(bbox[1]*416 - bbox[3]*416/2))),
                                    (int(float(bbox[0]*416 + bbox[2]*416/2)),
                                    int(float(bbox[1]*416 + bbox[3]*416/2))), (1.0, 0, 0), 2)
    lab = np.reshape(y_true1,[-1,3,15])
    bboxs = lab[lab[:,:,4] > 0.5][:,:4]
    for i in range(bboxs.shape[0]):
        bbox = bboxs[i]
        image = cv2.rectangle(image, (int(float(bbox[0]*416 - bbox[2]*416/2)),
                                    int(float(bbox[1]*416 - bbox[3]*416/2))),
                                    (int(float(bbox[0]*416 + bbox[2]*416/2)),
                                    int(float(bbox[1]*416 + bbox[3]*416/2))), (1.0, 0, 0), 2)
    plt.imshow(image)


#%%


#%%
if __name__ == '__main1__':
    annotation_path = r'D:\sysDef\Documents\GitHub\pytest\dataAndModel\data\mnist\objtrainlab.txt'
    anchors_path = 'yolo_anchors.txt'
    with open(annotation_path) as f:
        lines = f.readlines()
    anchors = get_anchors(anchors_path)
    gen = data_generator(annotation_lines=lines, batch_size=1,
                  input_shape=(416,416),
                  anchors=anchors, num_classes=10)
    for image_data, y_true0, y_true1 in gen:
        print(image_data.shape)
        print(y_true0.shape)
        print(y_true1.shape)
        

        
        print((np.reshape(y_true0,[-1,3,15])[:,:,4] > 0.5).sum())
        print((np.reshape(y_true1,[-1,3,15])[:,:,4] > 0.5).sum())

        import cv2
        from cv2 import cv2
        image = image_data[0].copy()


        lab = np.reshape(y_true0,[-1,3,15])
        bboxs = lab[lab[:,:,4] > 0.5][:,:4]
        for i in range(bboxs.shape[0]):
            bbox = bboxs[i]
            image = cv2.rectangle(image, (int(float(bbox[0]*416 - bbox[2]*416/2)),
                                        int(float(bbox[1]*416 - bbox[3]*416/2))),
                                        (int(float(bbox[0]*416 + bbox[2]*416/2)),
                                        int(float(bbox[1]*416 + bbox[3]*416/2))), (1.0, 0, 0), 2)
        lab = np.reshape(y_true1,[-1,3,15])
        bboxs = lab[lab[:,:,4] > 0.5][:,:4]
        for i in range(bboxs.shape[0]):
            bbox = bboxs[i]
            image = cv2.rectangle(image, (int(float(bbox[0]*416 - bbox[2]*416/2)),
                                        int(float(bbox[1]*416 - bbox[3]*416/2))),
                                        (int(float(bbox[0]*416 + bbox[2]*416/2)),
                                        int(float(bbox[1]*416 + bbox[3]*416/2))), (1.0, 0, 0), 2)
        plt.imshow(image)

        break
