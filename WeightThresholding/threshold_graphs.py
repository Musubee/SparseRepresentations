import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensornets as nets

import torchvision.models as models

from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16


keras = tf.keras

def extract_tf_weights(model):
	"""Return weights list from given TF model."""

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
	'''Return weights list from given Torch model.'''
	params = [p.data for p in model.parameters()]
	for p in model.parameters():
		params.append(p.data)
	params = [p.reshape(-1).numpy().tolist() for p in params] # now list of weights
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

num_weights = [len(alexnet_weights), len(vgg_weights), len(googlenet_weights), len(resnet_weights)]
index = np.arange(len(model_names))
plt.bar(index, num_weights, color = 'red', alpha = 0.5)
plt.xticks(index, model_names, fontsize = 7, rotation = 30)
plt.ylabel('Number of Variables', fontsize = 10)
plt.title('Number of Trainable Variables in Past ILSVRC Winners', fontsize = 10)
plt.show()


