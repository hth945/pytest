
#加载飞桨和相关类库
import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph.nn import Linear
import numpy as np
import os
from PIL import Image

# 如果～/.cache/paddle/dataset/mnist/目录下没有MNIST数据，API会自动将MINST数据下载到该文件夹下
# 设置数据读取器，读取MNIST数据训练集
trainset = paddle.dataset.mnist.train()
# 包装数据读取器，每次读取的数据数量设置为batch_size=8
train_reader = paddle.batch(trainset, batch_size=8)

# 以迭代的形式读取数据
for batch_id, data in enumerate(train_reader()):
    # 获得图像数据，并转为float32类型的数组
    img_data = np.array([x[0] for x in data]).astype('float32')
    # 获得图像标签数据，并转为float32类型的数组
    label_data = np.array([x[1] for x in data]).astype('float32')
    # 打印数据形状
    print("图像数据形状和对应数据为:", img_data.shape, img_data[0])
    print("图像标签形状和对应数据为:", label_data.shape, label_data[0])
    break

print("\n打印第一个batch的第一个图像，对应标签数字为{}".format(label_data[0]))
# 显示第一batch的第一个图像
import matplotlib.pyplot as plt
img = np.array(img_data[0]+1)*127.5
img = np.reshape(img, [28, 28]).astype(np.uint8)

plt.figure("Image") # 图像窗口名称
plt.imshow(img)
plt.axis('on') # 关掉坐标轴为 off
plt.title('image') # 图像题目
plt.show()


# 定义mnist数据识别网络结构，同房价预测网络
class MNIST(fluid.dygraph.Layer):
    def __init__(self):
        super(MNIST, self).__init__()

        # 定义一层全连接层，输出维度是1，激活函数为None，即不使用激活函数
        self.fc = Linear(input_dim=784, output_dim=1, act=None)

    # 定义网络结构的前向计算过程
    def forward(self, inputs):
        outputs = self.fc(inputs)
        return outputs

#
# # 通过with语句创建一个dygraph运行的context
# # 动态图下的一些操作需要在guard下进行
# with fluid.dygraph.guard():
#     model = MNIST()
#     model.train()
#     train_loader = paddle.batch(paddle.dataset.mnist.train(), batch_size=16)
#     optimizer = fluid.optimizer.SGDOptimizer(learning_rate=0.001, parameter_list=model.parameters())
#     EPOCH_NUM = 10
#     for epoch_id in range(EPOCH_NUM):
#         for batch_id, data in enumerate(train_loader()):
#             # 准备数据，格式需要转换成符合框架要求
#             image_data = np.array([x[0] for x in data]).astype('float32')
#             label_data = np.array([x[1] for x in data]).astype('float32').reshape(-1, 1)
#             # 将数据转为飞桨动态图格式
#             image = fluid.dygraph.to_variable(image_data)
#             label = fluid.dygraph.to_variable(label_data)
#
#             # 前向计算的过程
#             predict = model(image)
#
#             # 计算损失，取一个批次样本损失的平均值
#             loss = fluid.layers.square_error_cost(predict, label)
#             avg_loss = fluid.layers.mean(loss)
#
#             # 每训练了1000批次的数据，打印下当前Loss的情况
#             if batch_id != 0 and batch_id % 1000 == 0:
#                 print("epoch: {}, batch: {}, loss is: {}".format(epoch_id, batch_id, avg_loss.numpy()))
#
#             # 后向传播，更新参数的过程
#             avg_loss.backward()
#             optimizer.minimize(avg_loss)
#             model.clear_gradients()
#
#     # 保存模型
#     fluid.save_dygraph(model.state_dict(), 'mnist')



# 读取一张本地的样例图片，转变成模型输入的格式
def load_image(img_path):
    trainset = paddle.dataset.mnist.train()
    # 包装数据读取器，每次读取的数据数量设置为batch_size=8
    train_reader = paddle.batch(trainset, batch_size=8)

    # 以迭代的形式读取数据
    for batch_id, data in enumerate(train_reader()):
        # 获得图像数据，并转为float32类型的数组
        img_data = np.array([x[0] for x in data]).astype('float32')
        # 获得图像标签数据，并转为float32类型的数组
        label_data = np.array([x[1] for x in data]).astype('float32')
        # 打印数据形状
        print("图像数据形状和对应数据为:", img_data.shape, img_data[0])
        print("图像标签形状和对应数据为:", label_data.shape, label_data[0])
        break
    print("\n打印第一个batch的第一个图像，对应标签数字为{}".format(label_data[0]))
    im = img_data[0]
    # im = img_data[0].reshape((28,28))
    # print(im.shape)
    # im = im.resize((28, 28), Image.ANTIALIAS)
    # print(im.shape)
    # im = np.array(im).reshape(1, -1).astype(np.float32)
    # # 图像归一化，保持和数据集的数据范围一致
    # im = 1 - im / 127.5
    return im

# 定义预测过程
with fluid.dygraph.guard():
    model = MNIST()
    params_file_path = 'mnist'
    img_path = './work/example_0.png'
# 加载模型参数
    model_dict, _ = fluid.load_dygraph("mnist")
    model.load_dict(model_dict)
# 灌入数据
    model.eval()
    tensor_img = load_image(img_path)
    result = model(fluid.dygraph.to_variable(tensor_img))
#  预测输出取整，即为预测的数字，打印结果
    print("本次预测的数字是", result.numpy().astype('int32'))