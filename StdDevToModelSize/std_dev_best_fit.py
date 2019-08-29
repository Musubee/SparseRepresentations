import torchvision.models as models

import scipy.optimize
import scipy.stats
from scipy.stats import norm

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

def load_models():
	alexnet = models.alexnet(pretrained = True)
	vgg16 = models.vgg16(pretrained = True)
	googlenet = models.googlenet(pretrained = True)
	resnet18 = models.resnet18(pretrained = True)
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
	mobilenet_v2 = models.mobilenet_v2(pretrained = True)
	resnext50_32x4d = models.resnext50_32x4d(pretrained = True)
	resnext101_32x8d = models.resnext101_32x8d(pretrained = True)
	return [alexnet, vgg11, vgg13, vgg13_bn, vgg16, vgg16_bn, vgg19, vgg19_bn,
	 resnet18, resnet34, resnet50, resnet101, resnet152,
	 squeezenet1_0, squeezenet1_1, densenet121, densenet161, densenet169, densenet201,
	 inception_v3, googlenet, mobilenet_v2, resnext50_32x4d, resnext101_32x8d]

def extract_torch_weights(model):
	'''Extract weights from TorchVision model as list'''
	params = []
	for p in model.parameters():
		params.append(p.data)
	params = [p.reshape(-1).numpy().tolist() for p in params] # now list of list of weights
	weights = [w for p in params for w in p]
	return weights

# def find_best_lin_fit(X, Y):
# 	'''Directly computes linear best fit to points.'''
# 	x_avg = sum(X)/len(X)
# 	y_avg = sum(Y)/len(Y)
# 	numer = sum([x * y for x,y in zip(X,Y)]) - len(X) * x_avg * y_avg
# 	denom = sum([x**2 for x in X]) - len(X) * x_avg**2
# 	b = numer/denom
# 	a = y_avg - b * x_avg
# 	return a,b

def inv_lin(x, a, b, c):
	return a/(x + b) + c

def find_best_fit(X, Y):
	return scipy.optimize.curve_fit(inv_lin, X, Y)

models = load_models()

weight_len = [len(extract_torch_weights(model)) for model in models]
std_devs = [norm.fit(extract_torch_weights(model))[1] for model in models]

print(std_devs)
coeffs, _ = find_best_fit(weight_len, std_devs)

X = sorted(weight_len)
Y = [inv_lin(x, coeffs[0], coeffs[1], coeffs[2]) for x in X]
plt.scatter(weight_len, std_devs)
plt.plot(X,Y, 'r')
plt.title('Model Size vs Std Dev for TorchVision models')
plt.xlabel('Model Size (Number of Weights)')
plt.ylabel('Std Dev of Distribution of Model Weights')
plt.show()


