#%%
import _thread
import time
import numpy as np

xyflag = 0  # 0x01: 要向目标移动  0x02:舵机按一下
xygetToFlag = 0 # 到达目标结果
xyNow = np.array([0,0]) # 当前识别到的xy坐标有可能识别错误
xyTarget = np.array([0,0]) # 鼠标点到的坐标

# 向目标移动的线程 输入目标坐标,图像计算的当前坐标,是否移动

def xyRun(threadName, delay):
    while True:
        time.sleep(delay)
        print(xyflag)
        # if flag&0x02 != 0:
        #     # stopMove
        #     # 下
        #     # time.sleep(delay)
        #     # 上
        #     pass
        if xyflag&0x01 != 0:
            # move(xyNow-xyTarget)
            # if 目标到达 getToFlag =1
            pass

imgType = 0  # 图像型号 0表示都不是
imgxy = np.array([0,0])  # 识别出的结果(imgType为0时无效)
# 图像的识别
def imgRun(threadName, delay):
    while True:
        # img识别
        pass


def type1Run(tt):
    global xyflag
    global xygetToFlag
    # 等待画面类型正确

    #开始并等待移动完成
    if imgType == tt:
        xyNow = imgxy
        xyflag |= 0x01
        xygetToFlag = 0
    for i in range(10):
        time.sleep(0.1)  # 100ms
        if xygetToFlag == 1:
            break
        if i > 8:
            raise Exception('print(a)')

    #按下

    #等待按成功

targetDetection = 0
detection = 0
x = 0
y = 0

_thread.start_new_thread(xyRun, ("Thread-1", 2, ))
while 1:
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


