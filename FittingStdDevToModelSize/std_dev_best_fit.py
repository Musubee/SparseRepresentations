import torchvision.models as models

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

def find_best_fit(X, Y):
	'''Directly computes linear best fit to points.'''
	x_avg = sum(X)/len(X)
	y_avg = sum(Y)/len(Y)
	numer = sum([x * y for x,y in zip(X,Y)]) - len(X) * x_avg * y_avg
	denom = sum([x**2 for x in X]) - len(X) * x_avg**2
	b = numer/denom
	a = y_avg - b * x_avg
	return a,b

alexnet = models.alexnet(pretrained = True)
vgg = models.vgg16(pretrained = True)
googlenet = models.googlenet(pretrained = True)
resnet18 = models.resnet18(pretrained = True)
models = [alexnet, vgg, googlenet, resnet18]

weight_len = [len(extract_torch_weights(model)) for model in models]
std_devs = [0.013, 0.009, 0.06, 0.029]

a,b = find_best_fit(weight_len, std_devs)
plt.scatter(weight_len, std_devs)
y_fit = [a + b * weight for weight in weight_len]
plt.plot(weight_len, y_fit)
plt.show()


