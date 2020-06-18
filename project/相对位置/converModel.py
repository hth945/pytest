import os
import time
import shutil
import numpy as np
import tensorflow as tf
from config import cfg
import cv2
import mynet


def conver2resizeInput2pb():

	# 从原来的h5文件中提取出只保存权重的文件
	modelTrain = tf.keras.models.load_model('oldModel.h5')
	modelTrain.save_weights('oldModelWeights.h5')

	# 建立一个除了输入尺寸，结构层数和以前相同的网络用来加载权重
	size =3648#  cfg.TEST.INPUT_SIZE
	# inputs = tf.keras.Input(shape=(3648, 5472, 3), name='in')
	inputs = tf.keras.Input(shape=(size, size, 3), name='in')
	imgDeconv, imgLab = mynet.Mynet(inputs,train=False)
	modelW = tf.keras.Model(inputs,  [imgDeconv, imgLab])
	modelW.load_weights('oldModelWeights.h5')

	#  建立一个 和 modelW共享权重，但是只输出lab的 用来训练的网络
	network = tf.keras.Model(inputs, imgLab)
	network.save("testmodel.h5")

	img = np.zeros([1, size, size, 3], dtype=np.uint8)
	img2 = img.astype(np.float32)
	label = network.predict(img2)

	# Convert Keras model to ConcreteFunction
	full_model = tf.function(lambda x: network(x))
	full_model = full_model.get_concrete_function(
	tf.TensorSpec(network.inputs[0].shape, network.inputs[0].dtype))

	# Get frozen ConcreteFunction
	from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
	frozen_func = convert_variables_to_constants_v2(full_model)
	frozen_func.graph.as_graph_def()

	layers = [op.name for op in frozen_func.graph.get_operations()]
	print("-" * 50)
	print("Frozen model layers: ")
	for layer in layers:
		print(layer)

	print("-" * 50)
	print("Frozen model inputs: ")
	print(frozen_func.inputs)
	print("Frozen model outputs: ")
	print(frozen_func.outputs)

	# Save frozen graph from frozen ConcreteFunction to hard drive
	tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
			logdir="./frozen_models",
			name="frozen_graph.pb",
			as_text=False)


def conver2resizeInput():

	# 从原来的h5文件中提取出只保存权重的文件
	modelTrain = tf.keras.models.load_model('oldModel.h5')
	modelTrain.save_weights('oldModelWeights.h5')

	# 建立一个除了输入尺寸，结构层数和以前相同的网络用来加载权重
	size =3648#  cfg.TEST.INPUT_SIZE
	inputs = tf.keras.Input(shape=(3648, 5472, 3), name='in')
	# inputs = tf.keras.Input(shape=(size, size, 3), name='in')
	imgDeconv, imgLab = mynet.Mynet(inputs,train=False)
	modelW = tf.keras.Model(inputs,  [imgDeconv, imgLab])
	modelW.load_weights('oldModelWeights.h5')

	#  建立一个 和 modelW共享权重，但是只输出lab的 用来训练的网络
	network = tf.keras.Model(inputs, imgLab)
	network.save("testmodel.h5")
	print(network.summary())

	img = np.zeros([1, 3648, 5472, 3], dtype=np.uint8)
	img2 = img.astype(np.float32)
	label = network.predict(img2)
	print(label.shape)


conver2resizeInput()


