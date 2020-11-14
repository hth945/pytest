# 导入图像读取第三方库
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image
import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph.nn import Linear

# 读取图像
img1 = cv2.imread('./work/example_0.png')
example = mpimg.imread('./work/example_0.png')
# 显示图像
plt.imshow(example)
plt.show()
im = Image.open('./work/example_0.png').convert('L')
print(np.array(im).shape)
im = im.resize((28, 28), Image.ANTIALIAS)
plt.imshow(im)
plt.show()
print(np.array(im).shape)


