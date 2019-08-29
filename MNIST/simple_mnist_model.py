import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torchsummary import summary

from torch.autograd import Variable
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import scipy.stats
from scipy.stats import norm

import random
import math

# taken from https://www.kaggle.com/justuser/mnist-with-pytorch-fully-connected-network
# with a few changes to data processing and network architecture
# Similar model achieves 4.7% error rate according to LeCun et al. 1998

# Hyper parameters: 
input_size = 784
output_size = 10
hidden_size = 500

epochs = 5
batch_size = 1
learning_rate = 0.00005

class Identity(nn.Module):
	#https://discuss.pytorch.org/t/how-to-delete-layer-in-pretrained-model/17648/2
	def __init__(self, x=0, dim=0):
		super(Identity, self).__init__()

	def forward(self, x):
		return x

class Simple_MNIST_Model(nn.Module):

	def __init__(self):
		super(Simple_MNIST_Model, self).__init__()
		self.l1 = nn.Linear(input_size, hidden_size, bias = False)
		self.l2 = nn.Linear(hidden_size, output_size, bias = False)
		self.softmax = nn.Softmax(dim = 0)

	def forward(self, x):
		x = self.l1(x)
		#return self.l2(x)
		x = self.l2(x)
		x = self.softmax(x)
		return x

def get_mnist_data():
	mnist_train_data = datasets.MNIST('dataset/', train = True, transform = transforms.ToTensor())
	mnist_train_data = [(torch.reshape(sample[0], [784]), sample[1]) for sample in mnist_train_data]
	random.shuffle(mnist_train_data)

	mnist_test_data = datasets.MNIST('dataset/', train = False, transform = transforms.ToTensor())
	mnist_test_data = [(torch.reshape(sample[0], [784]), sample[1]) for sample in mnist_test_data]
	return mnist_train_data, mnist_test_data

def define_model():
	net = Simple_MNIST_Model()

	optimizer = torch.optim.SGD(net.parameters(), lr = learning_rate, momentum = 0.9)
	loss_func = F.mse_loss

	return net, optimizer, loss_func

def train_model(net, optimizer, loss_func, train_data):
	print('training')
	for e in range(epochs):
		for sample in train_data:
			x = sample[0]
			y = sample[1]

			x_var = Variable(x)
			y_var = torch.zeros([10], dtype = torch.float)
			y_var[y] = 1

			optimizer.zero_grad()
			out = net(x_var)

			loss = loss_func(out, y_var)
			loss.backward()
			optimizer.step()

		print('Epoch: {}, Loss Data: {}'.format(e, loss.data))

def get_model_weights(net):
	parameters = []
	for p in net.parameters():
		parameters.append(p.data)
	parameters = [p.reshape(-1).numpy().tolist() for p in parameters]
	weights = [w for p in parameters for w in p]
	return weights

def plot_weight_histogram(weights):
	iqr = scipy.stats.iqr(weights)
	bin_width = (2 * iqr)/(len(weights) ** (1/3))

	# Freedman-Diaconis Rule 
	num_bins = int(round(((max(weights) - min(weights)) / bin_width)))

	n, bins, patches = plt.hist(weights, bins = num_bins, density = True)

	(mu, sigma) = norm.fit(weights)

	y = scipy.stats.norm.pdf(bins, mu, sigma)
	l = plt.plot(bins, y, 'r--', linewidth = 1)

	plt.title(r'$\mathrm{Histogram\ of\ MNIST\ Model\ Weights:}\ \mu=%.3f,\ \sigma=%.3f$' %(mu, sigma))
	plt.show()

def remove_softmax(net):
	# Removes softmax layer after training so raw outputs can be analyzed
	net.softmax = Identity()

def remove_softmax_layer2(net):
	# Removes last two layers so output of first hidden layer is computed
	net.l2 = Identity()
	net.softmax = Identity()

def get_sample_test_set(test_data):
	# Returns one sample from each class 
	used = {}
	for sample in test_data:
		if sample[1] not in used:
			used[sample[1]] = sample
	return list(used.values())

def get_correct_samples(net, test_data):
	used = {}
	for sample in test_data:
		# AND correct
		if sample[1] not in used:
			x = sample[0]
			y = sample[1]
			x_var = Variable(x)

			out = net(x_var)
			out_array = np.asarray([element.item() for element in out.flatten()])
			max_index = np.argmax(out_array)
			if y == max_index:
				used[y] = sample
	return list(used.values())

def sample_test_outputs(net, test_data):
	# returns outputs of nets on sample test data
	outputs = []
	for sample in test_data:
		x = sample[0]
		x_var = Variable(x)
		outputs.append((net(x_var), sample[1]))

	return outputs

def plot_test_outputs(outs):
	x = range(10)
	for output in outs:
		label = output[1]
		tensor = output[0]
		y = tensor.tolist()
		plt.bar(x, y, alpha = 0.5)
		plt.title('Output for Class ' + str(label))
		plt.xlabel('Class')
		plt.show()
		plt.clf()

def plot_hidden_layer_hist(outs):
	for output in outs:
		label = output[1]
		tensor = output[0]
		hidden_outs = tensor.tolist()
		iqr = scipy.stats.iqr(hidden_outs)
		bin_width = (2 * iqr)/(len(hidden_outs) ** (1/3))

		# Freedman-Diaconis Rule
		num_bins = int(round(((max(hidden_outs) - min(hidden_outs)) / bin_width)))

		n, bins, patches = plt.hist(hidden_outs, bins = num_bins, density = True)

		mu, sigma = norm.fit(hidden_outs)
		y = norm.pdf(bins, mu, sigma)
		l = plt.plot(bins, y, 'r--', linewidth = 1)

		plt.title('Histogram of 1st Hidden Layer Outputs  for Class {}: mu = {}, sigma = {}'.format(label, mu, sigma))
		plt.show()

def kl_divergence(m1, s1, m2, s2):
	term1 = math.log2(s2/s1)
	term2 = ((s1**2 + (m1 - m2)**2) / (2 * (s2**2)))
	term3 = 0.5
	return term1 + term2 - term3

def get_theoretical_hidden_out(input_vector, m, s):
	inputs = input_vector.tolist()
	input_sum = sum(inputs)
	input_sum_sq = sum([i**2 for i in inputs])
	mu = input_sum * m
	sigma = (input_sum_sq**0.5)*s

	return mu, sigma
	

mnist_train_data, mnist_test_data = get_mnist_data()

net, optimizer, loss_func = define_model()

train_model(net, optimizer, loss_func, mnist_train_data)

weights = get_model_weights(net)

mu, sigma = norm.fit(weights)

correct_samples = get_correct_samples(net, mnist_test_data)

remove_softmax_layer2(net)

net_outs = sample_test_outputs(net, correct_samples)

hidden_outs = [out[0].tolist() for out in net_outs]

for i, out_vector in enumerate(hidden_outs):
	input_vector = correct_samples[i][0]
	m1, s1 = get_theoretical_hidden_out(input_vector, mu, sigma)
	m2, s2 = norm.fit(out_vector)
	divergence = kl_divergence(m1, s1, m2, s2)
	print('Class: {}, KL Divergence: {}'.format(correct_samples[i][1], divergence))


# plot_hidden_layer_hist(net_outs)



# weights = get_model_weights(net)
# plot_weight_histogram(weights)
# remove_softmax(net)

# sample_test_data = get_sample_test_set(mnist_test_data)
# net_outs = sample_test_outputs(net, sample_test_data)

# plot_test_outputs(net_outs)



# print('testing')

# total_correct = 0
# num_samples = len(mnist_test_data)
# for sample in mnist_test_data:
# 	x = sample[0]
# 	y = sample[1]
# 	x_var = Variable(x)

# 	out = net(x_var)
# 	out_array = np.asarray([element.item() for element in out.flatten()])
# 	print(out_array)

# 	max_index = np.argmax(out_array)
# 	if y == max_index:
# 		total_correct += 1

# print(total_correct/num_samples)
	





