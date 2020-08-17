#%%
import os
import cv2
import numpy as np
import time
import random

def get_screenshot(id):
    os.system('adb shell screencap -p /sdcard/%s.png' % str(id))
    os.system('adb pull /sdcard/%s.png .' % str(id))

def jump(distance):
    # 这个参数还需要针对屏幕分辨率进行优化
    press_time = int(distance * 1.35)

    # 生成随机手机屏幕模拟触摸点
    # 模拟触摸点如果每次都是同一位置，成绩上传可能无法通过验证
    rand = random.randint(0, 9) * 10
    cmd = ('adb shell input swipe %i %i %i %i ' + str(press_time)) \
          % (320 + rand, 410 + rand, 320 + rand, 410 + rand)
    os.system(cmd)
    print(cmd)

get_screenshot(0)
# %%
