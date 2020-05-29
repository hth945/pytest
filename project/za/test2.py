#%%
import _thread
import time
import numpy as np

xyflag = 0  # 0x01: 要向目标移动  0x02:舵机按一下
xygetToFlag = 0 # 到达目标结果
xyTarget = np.array([0,0]) # 鼠标点到的坐标


imgType = 0  # 图像型号 0表示都不是
imgxy = np.array([0, 0])  # 识别出的结果(imgType为0时无效)
# 图像的识别
def imgRun(threadName, delay):
    global xyflag
    global xygetToFlag
    while True:
        # img识别

        if imgType == xyTarget: # 识别的结果却是是
            if xyflag & 0x01 != 0:
                xyflag &= ~0x01
                # move(xyNow-xyTarget)
                # if 目标到达 getToFlag =1
        else:
            print("imgType != xyTarget")


def type1Run(tt):
    global xyflag
    global xygetToFlag
    #开始并等待移动完成
    if imgType == tt:
        xyNow = imgxy
        xygetToFlag = 0
        xyflag |= 0x01
    else:
        raise Exception('print(a)')
    for i in range(100):
        time.sleep(0.1)  # 100ms
        if xygetToFlag == 1:
            break
        if i > 8:
            raise Exception('10s 没有移动到位置')

    #按下

    #等待按成功

targetDetection = 0
detection = 0
x = 0
y = 0

_thread.start_new_thread(xyRun, ("Thread-1", 2, ))
while True:
    str = input("in:")
    if str == "n":
       flag |= 0x01
    elif str == "l":
       flag &= ~0x01

    #识别结果 Detection x,y
    if detection == targetDetection: # 已经切换到目标图像
        if detection == 0:  # 初始要按开始
            if getToFlag == 1: # 到达目标
                flag &= ~0x01
                # 下
                # time.sleep(delay)
                # 上
   print(str)


