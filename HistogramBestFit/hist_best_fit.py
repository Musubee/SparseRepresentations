import torchvision.models as models

import pandas as pd
import scipy.stats
from scipy.stats import norm

import numpy as np


import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

def extract_torch_weights(model):
	'''Extract weights from TorchVision model as list'''
	params = []
	for p in model.parameters():
		params.append(p.data)
	params = [p.reshape(-1).numpy().tolist() for p in params] # now list of trainable vars
	weights = [w for p in params for w in p]
	return weights



def plot_hist_best_fit(model, model_name):
	model_weights = extract_torch_weights(model)
	iqr = scipy.stats.iqr(model_weights)
	bin_width = (2 * iqr)/(len(model_weights) ** (1/3))

	# Freedman-Diaconis Rule
	num_bins = int(round(((max(model_weights) - min(model_weights)) / bin_width)))

	n, bins, patches = plt.hist(model_weights, bins = num_bins, density = True)

	(mu, sigma) = norm.fit(model_weights)
	y = scipy.stats.norm.pdf(bins, mu, sigma)
	l = plt.plot(bins, y, 'r--', linewidth = 1)

	plt.title(r'$\mathrm{Histogram\ of\ %s\ Weights:}\ \mu=%.3f,\ \sigma=%.3f$' %(model_name, mu, sigma))
	plt.show()

def plot_num_weights(model_info):
	model_weight_info = [(len(extract_torch_weights(model[0])), model[1]) for model in model_info]
	num_weights = [model[0] for model in model_weight_info]
	model_names = [model[1] for model in model_weight_info]
	index = np.arange(len(model_names))
	plt.bar(index, num_weights, color = 'red', alpha = 0.5)
	plt.xticks(index, model_names, fontsize = 7, rotation = 30)
	plt.ylabel('Number of Weights', fontsize = 10)
	plt.title('Number of Weights in Popular Computer Vision Models', fontsize = 10)
	plt.show()

# Models
# alexnet = models.alexnet(pretrained = True)
vgg11 = models.vgg11(pretrained = True)
vgg13 = models.vgg13(pretrained = True)
vgg13_bn = models.vgg13_bn(pretrained = True)
vgg16_bn = models.vgg16_bn(pretrained = True)
vgg19 = models.vgg19(pretrained = True)
vgg19_bn = models.vgg19(pretrained = True)
resnet34 = models.resnet34(pretrained = True)
resnet50 = models.resnet50(pretrained = True)
resnet101 = models.resnet101(pretrained = True)
resnet152 = models.resnet152(pretrained = True)
squeezenet1_0 = models.squeezenet1_0(pretrained = True)
squeezenet1_1 = models.squeezenet1_1(pretrained = True)
densenet121 = models.densenet121(pretrained = True)
densenet161 = models.densenet161(pretrained = True)
densenet169 = models.densenet169(pretrained = True)
densenet201 = models.densenet201(pretrained = True)
inception_v3 = models.inception_v3(pretrained = True)
shufflenet_v2_x0_5 = models.shufflenet_v2_x0_5(pretrained = True)
shufflenet_v2_x1_0 = models.shufflenet_v2_x1_0(pretrained = True)
shufflenet_v2_x1_5 = models.shufflenet_v2_x1_5(pretrained = True)
shufflenet_v2_x2_0 = models.shufflenet_v2_x2_0(pretrained = True)
mobilenet_v2 = models.mobilenet_v2(pretrained = True)
resnext50_32x4d = models.resnext50_32x4d(pretrained = True)
resnext101_32x8d = modelos.resnext101_32x8d(pretrained = True)


# vgg = models.vgg16(pretrained = True)
# googlenet = models.googlenet(pretrained = True)
# resnet18 = models.resnet18(pretrained = True)

# model_info = [(alexnet, 'AlexNet'), (vgg, 'VGG16'), (googlenet, 'GoogLeNet'), (resnet18, 'ResNet18')]

# plot_num_weights(model_info)

