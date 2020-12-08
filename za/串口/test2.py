import numpy as np
import cv2
import serial
import time
import _thread
import threading
import screeninfo

lock=threading.Lock() #申请一把锁
def runCmd(serialX, cmd):
    lock.acquire()
    serialX.write((cmd + '\n').encode(encoding='ASCII', errors='strict'))
    temData = serialX.readline()
    lock.release()
    return str(temData[:-1], encoding="utf-8")


serial1 = serial.Serial('COM7', 115200, timeout=0.5)  #/dev/ttyUSB0
runCmd(serial1, 'connection')

def imgRun():




    while True:
        time.sleep(1)


onOff = 0
showN=0

screen_id = 0
is_color = True

# get the size of the screen
screen = screeninfo.get_monitors()[screen_id]
width, height = screen.width, screen.height
width = 3840
height = 2160

imgs=[]
image = np.zeros((height, width, 3), dtype=np.float32)
imgs.append(image)
image = np.ones((height, width, 3), dtype=np.float32)
imgs.append(image)

image = np.ones((height, width, 3), dtype=np.float32)
tem = int(height/3)
image[:tem,:]=[1.0,0,0]
image[tem:tem*2, :] = [0, 1.0, 0]
image[tem*2 :, :] = [0, 0, 1.0]
imgs.append(image)



window_name = 'projector'
cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
cv2.moveWindow(window_name, screen.x - 1, screen.y - 1)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
cv2.imshow(window_name, imgs[showN%len(imgs)])
cv2.waitKey(1)

_thread.start_new_thread(imgRun,())
while True:
    key = runCmd(serial1, 'getKey')
    if key!='0':
        print(key)
        if key=='1':
            if onOff == 0:
                onOff = 1
                runCmd(serial1, 'setVddValue1:3.3')
                runCmd(serial1, 'setVddValue2:12')
                runCmd(serial1, 'setVddOnOff1:1')
                runCmd(serial1, 'setVddOnOff2:1')
            else:
                onOff = 0
                runCmd(serial1, 'setVddOnOff1:0')
                runCmd(serial1, 'setVddOnOff2:0')
            print('onOff: ',onOff)
        if key=='2':
            showN += 1
            cv2.imshow(window_name, imgs[showN % len(imgs)])
            cv2.waitKey(1)
            print('showN',showN)
        if key=='4':
            showN -= 1
            cv2.imshow(window_name, imgs[showN % len(imgs)])
            cv2.waitKey(1)
            print('showN', showN)
    else:
        time.sleep(0.01)
        