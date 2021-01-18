import os
import time
import datetime
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cv2
import numpy as np
import sys
# sys.path.insert(0,r'D:\ChangZhi\dnplayer2')

os.environ['Path']=os.environ['Path']+';D:\ChangZhi\dnplayer2'
os.system('set path')
def get_screenshot(id=0):
    os.system('adb shell screencap -p /sdcard/%s.png' % str(0))
    os.system('adb pull /sdcard/0.png ./%s.png ' % str(id))
    img = cv2.imread('./%s.png'  % str(id))
    return img

def adbClick(x,y):
    cmd = ('adb shell input swipe %i %i %i %i ' ) % (x, y, x, y)
    os.system(cmd)
    print(cmd)

idSave = 2
def on_click(event):
    dst_x, dst_y = event.xdata, event.ydata
    print(dst_x, dst_y)

    img = get_screenshot()
    im.set_array(img)
    global idSave
    cv2.imwrite('src%d.png'%(idSave), img)
    idSave += 1

    if dst_x != None:
        adbClick(dst_x,dst_y)


fig = plt.figure()
img = get_screenshot() # np.zeros([100,100],dtype=np.uint8)
print(img)
im = plt.imshow(img, animated=True)
fig.canvas.mpl_connect('button_press_event', on_click)
plt.show()


