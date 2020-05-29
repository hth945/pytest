import numpy as np
import cv2
import serial
import time
import _thread
import threading

lock=threading.Lock() #申请一把锁
def runCmd(serialX, cmd):
    lock.acquire()
    serialX.write((cmd + '\n').encode(encoding='ASCII', errors='strict'))
    temData = serialX.readline()
    lock.release()
    # print(temData)

def ServoSet(serialX, d):
    if d < 0:
        d = 0
    if d > 1:
        d = 1
    runCmd(serialX, 'setServoC6:%d'%(499+d*2000))

def MotorSet(serialX, m1, m2, t):
    if m1 > 0:
        runCmd(serialX, 'setPinE8:1')
    elif m1 < 0:
        m1 = -m1
        runCmd(serialX, 'setPinE8:0')

    if m2 > 0:
        runCmd(serialX, 'setPinE9:1')
    elif m2 < 0:
        m2 = -m2
        runCmd(serialX, 'setPinE9:0')
    time.sleep(0.01)
    if m1 > 10000:
        m1 = 10000
    if m2 > 10000:
        m2 = 10000
    runCmd(serialX, 'setmotorB8:%d' % m1)
    runCmd(serialX, 'setmotorB9:%d' % m2)

    if (t > 0):
        time.sleep(t)
        runCmd(serialX, 'setmotorB8:0')
        runCmd(serialX, 'setmotorB9:0')
    print(m1, m2)

imgType = 0  # 图像型号 0表示都不是
imgTargetType = 0
xyflag = 0  # 0x01: 要向目标移动
xygetToFlag = 0 # 到达目标结果
xyTarget = np.array([0,0]) # 鼠标点到的坐标

def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
    global xyflag
    if event == cv2.EVENT_LBUTTONDOWN:
        xyflag = np.array([x,y])
        print(xyflag)

def imgRun():
    global xyflag
    global xygetToFlag
    global xyTarget
    global imgType
    global imgTargetType

    cv2.namedWindow("img2")
    cv2.setMouseCallback("img2", on_EVENT_LBUTTONDOWN)
    cv2.namedWindow("img")
    cv2.setMouseCallback("img", on_EVENT_LBUTTONDOWN)
    cv2.namedWindow("frame")
    cv2.setMouseCallback("frame", on_EVENT_LBUTTONDOWN)

    cap = cv2.VideoCapture(0)  # 创建内置摄像头变量
    cap.set(cv2.CAP_PROP_EXPOSURE,-5)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    red_lower = np.array([0,124,107])
    red_uper = np.array([14,255,255])

    key = cv2.waitKey(10) & 0xFF
    i = -1
    while(key != ord('q')):
        key = cv2.waitKey(10) & 0xFF
        if key == ord('u'):
            ServoSet(serial1, 0)
        if key == ord('d'):
            ServoSet(serial1, 0.6)
        if key == ord('n'):
            i = 4
        if key == ord('l'):
            MotorSet(serial1, 0, 0, 0)
            i = -1

        ret, frame = cap.read()  # 把摄像头获取的图像信息保存之img变量
        # frame_=cv2.GaussianBlur(frame,(5,5),0)
        hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        mask=cv2.inRange(hsv,red_lower,red_uper)

        mask=cv2.erode(mask, None, iterations=2)
        mask=cv2.dilate(mask, None, iterations=2)

        cnts=cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

        img = frame.copy()
        # cv2.drawContours(img, cnts, -1, (255, 255, 255), 3) #画所有边界
        img2 = cv2.bitwise_and(img,img,mask=mask)  #过滤边界外的图像
        cv2.imshow("img2", img2)
        cv2.imshow("frame", frame)

        if len(cnts) > 0:
            cnt = max (cnts,key=cv2.contourArea)  # 面积
            (color_x,color_y),color_radius=cv2.minEnclosingCircle(cnt)  #  最小外接圆
            if color_radius > 3:
                # 将检测到的颜色标记出来
                cv2.circle(img,(int(color_x),int(color_y)),int(color_radius),(255,0,255),2)  #画圆
                cv2.imshow("img", img)

                if imgType == imgTargetType and xyflag & 0x01 != 0:  # 图片整体类型正确且 需要向目标移动
                    i -= 1
                    if i == 0:
                        i = 4
                        effector = np.array([color_x, color_y])
                        dxy = xyTarget - effector
                        print('dxy: ', dxy)
                        if abs(dxy[0]) < 3:
                            dxy[0] = 0
                        if abs(dxy[1]) < 3:
                            dxy[1] = 0
                        if dxy[0] == 0 and dxy[1] == 0: # 已经到达目标位置
                            xygetToFlag = 1
                        else:
                            MotorSet(serial1, dxy[0] * 100, dxy[1] * 100, 0)


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


serial1 = serial.Serial('COM6', 115200, timeout=0.5)  #/dev/ttyUSB0
runCmd(serial1, 'connection')
MotorSet(serial1, 0, 0, 1)
# ServoSet(serial1, 0)
# MotorSet(serial1, 20000, 20000, 1)

_thread.start_new_thread(imgRun,())

while True:
    pass


