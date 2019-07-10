import os
import ssl

import tensorflow as tf
import tensornets as nets

import torchvision.models as models

from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16


import numpy as np
import matplotlib.pyplot as plt


# Models to gather: AlexNet (torch), ZF (tensornets), VGG (torch), GoogleLeNet(torch), Resnet (keras.apps)
# Models' Num Weights:
# AlexNet: 6.1e7
# VGG: 1.4e8
# GoogLeNet: 6.6e6
# ResNet: 2.6e7
keras = tf.keras

def extract_tf_weights(model):
	"""Return weights list from given model. Used with Keras applications."""

	init = tf.global_variables_initializer()
	with tf.Session() as sess:
		sess.run(init)

		variables = model.trainable_weights # returns TF variable list
		tensors = [variable.value() for variable in variables] # now TF tensors
		tensors = [tensor.eval(session = sess) for tensor in tensors] # now numpy arrays

		flat_tensors = [weight.flatten() for weight in tensors]
		weights = [weight for tensor in flat_tensors for weight in tensor]
		return weights

def extract_torch_weights(model):
	params = []
	for p in model.parameters():
		params.append(p.data)
	params = [p.reshape(-1).numpy().tolist() for p in params] # now list of trainable vars
	weights = [w for p in params for w in p]
	return weights

def get_ratio_thresholded_weights(weights, t):
	"""
	Returns fraction of weights under given threshold

	Arguments:
		weights (list of floats): weight list to be thresholded
		t (float): threshold value

	Returns:
		float: fraction of weights under t

	"""

	total_num_weights = len(weights)
	num_thresholded_weights = sum(abs(weight) <= t for weight in weights)

	return num_thresholded_weights/total_num_weights

def make_threshold_graph(model_names, model_weights, t):
	percentage_thresholded = [get_ratio_thresholded_weights(weights, t) for weights in model_weights]

	index = np.arange(len(model_names))
	plt.bar(index, percentage_thresholded, alpha = 0.5)
	plt.xlabel('Models', fontsize = 10)
	plt.ylabel('Percentage of Weights Under Threshold', fontsize = 10)
	plt.xticks(index, model_names, fontsize = 7, rotation = 30)
	plt.title('Popular Model Weights Under Threshold (t = ' + str(t) + ')')
	plt.show()


# inputs = tf.placeholder(tf.float32, [1, 224, 224, 3])
# model = nets.ZF(inputs)
# with tf.Session() as sess:
# 	init = tf.global_variables_initializer()
# 	sess.run(init)
# 	sess.run(model.pretrained())
# 	print(model.eval())

# Models
alexnet = models.alexnet(pretrained = True)
vgg = models.vgg16(pretrained = True)
googlenet = models.googlenet(pretrained = True)
resnet = ResNet50(weights = 'imagenet')

# Model Weights
alexnet_weights = extract_torch_weights(alexnet)
vgg_weights = extract_torch_weights(vgg)
googlenet_weights = extract_torch_weights(googlenet)
resnet_weights = extract_tf_weights(resnet)

models = [alexnet, vgg, googlenet, resnet]
model_names = ['AlexNet', 'VGG', 'GoogleLeNet', 'ResNet']
model_weights = [alexnet_weights, vgg_weights, googlenet_weights, resnet_weights]


# t_values = [1, 0.1, 0.01, 0.001]
# for t in t_values:
# 	print(t)
# 	make_threshold_graph(model_names, model_weights, t)

# for model in model_weights:
# 	print(len(model))

num_weights = [61100840, 138357544, 6624904, 25583592]
index = np.arange(len(model_names))
plt.bar(index, num_weights, color = 'red', alpha = 0.5)
plt.xticks(index, model_names, fontsize = 7, rotation = 30)
plt.ylabel('Number of Variables', fontsize = 10)
plt.title('Number of Trainable Variables in Past ILSVRC Winners', fontsize = 10)
plt.show()


