#%%

import pyautogui
im1 = pyautogui.screenshot()
im2 = pyautogui.screenshot('my_screenshot.png')

# %%
import pyautogui
img = pyautogui.screenshot(region=(0,0, 300, 400))

import os
from PIL import Image
import matplotlib.pyplot as plt
plt.figure("Image") # 图像窗口名称
plt.imshow(img)
plt.axis('on') # 关掉坐标轴为 off
plt.title('image') # 图像题目
plt.show()

# %%
import pyautogui

button7location = pyautogui.locateOnScreen('1.png')
print(button7location)
print(button7location[0])

print(button7location.left)

button7point = pyautogui.center(button7location)
print(button7point)

print(button7point[0])
print(button7point.x)
button7x, button7y = button7point
pyautogui.click(button7x, button7y)  # clicks the center of where the 7 button was found
pyautogui.click('calc7key.png') # a shortcut version to click on the center of where the 7 button was found

# %%
