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
import matplotlib.patches as mpatches
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
		self.dropout = nn.Dropout()
		self.l1 = nn.Linear(input_size, hidden_size)
		self.l2 = nn.Linear(hidden_size, output_size)
		self.softmax = nn.Softmax(dim = 0)

	def forward(self, x):
		x = self.dropout(x)
		x = self.l1(x)
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

def get_correct_samples(net, test_data, n):
	used = {}
	for sample in test_data:
		x = sample[0]
		y = sample[1]
		x_var = Variable(x)
		if y not in used:
			out = net(x_var)
			out_array = np.asarray([element.item() for element in out.flatten()])
			max_index = np.argmax(out_array)
			if y == max_index:
				used[y] = [sample]

		elif y in used and len(used[sample[1]]) < n:
			out = net(x_var)
			out_array = np.asarray([element.item() for element in out.flatten()])
			max_index = np.argmax(out_array)
			if y == max_index:
				used[y].append(sample)

	return list(used.values())

def sample_test_outputs(net, samples):
	# returns outputs of nets on sample test data
	outputs = []
	for label in samples:
		label_outs = []
		for sample in label:
			x = sample[0]
			x_var = Variable(x)
			label_outs.append((net(x_var), sample[1]))
		outputs.append(label_outs)

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

def get_fits(net_outs):
	fits = {}
	for label_outs in net_outs:
		label = label_outs[0][1]
		fits[label] = [norm.fit(out[0].tolist()) for out in label_outs]
	return fits

def average_fits(fits):
	for label in fits:
		outs = fits[label]
		mu_avg = (sum([out[0] for out in outs]))/(len(outs))
		sigma_avg = (sum([out[1] for out in outs]))/(len(outs))
		fits[label] = (mu_avg, sigma_avg)
	return fits

def cross_kl_divergence_experiment_actual(fits):
	kl_divs = {}
	for label in fits:
		for other_label in fits:
			m1 = fits[label][0]
			s1 = fits[label][1]
			m2 = fits[other_label][0]
			s2 = fits[other_label][1]

			kl_divs[(label, other_label)] = kl_divergence(m1, s1, m2, s2)

	keys = list(kl_divs.keys())
	keys.sort(key = lambda x: (x[0], x[1]))

	for label in keys:
		print(label)
		print(kl_divs[label])

def cross_kl_divergence_experiment_theoretical(samples, weights):
	# samples: list of list of input vectors 


	dist_dict = {}
	m_weights, s_weights = norm.fit(weights)
	for i, label_inputs in enumerate(samples):
		label = label_inputs[0][1]
		input_vectors = [x[0] for x in label_inputs]
		input_vector = torch.div(sum(input_vectors), len(input_vectors))
		dist_dict[label] = get_theoretical_hidden_out(input_vector, m_weights, s_weights)

	keys = list(dist_dict.keys())
	keys.sort()
	for key in keys:
		for other_key in keys:
			print('({}, {})'.format(key, other_key))
			print(kl_divergence(dist_dict[key][0], dist_dict[key][1], dist_dict[other_key][0], dist_dict[other_key][1]))

def intraclass_kl_divergence_experiment(samples, weights, fits):
	theoretical_distributions = {}
	m_weights, s_weights = norm.fit(weights)
	for i, label_inputs in enumerate(samples):
		# Theoretical distribution
		label = label_inputs[0][1]
		input_vectors = [x[0] for x in label_inputs]
		input_vector = torch.div(sum(input_vectors), len(input_vectors))
		theoretical_distributions[label] = get_theoretical_hidden_out(input_vector, m_weights, s_weights)

	for label in range(10):
		print('Class {}'.format(label))
		print('KL(theoretical, actual) = {}'.format(kl_divergence(theoretical_distributions[label][0], 
																	theoretical_distributions[label][1],
																	fits[label][0], fits[label][1])))

def inter_kl_div_actual_vs_theoretical(samples, weights, fits):
	theoretical_distributions = {}
	m_weights, s_weights = norm.fit(weights)
	for i, label_inputs in enumerate(samples):
		# Theoretical distribution
		label = label_inputs[0][1]
		input_vectors = [x[0] for x in label_inputs]
		input_vector = torch.div(sum(input_vectors), len(input_vectors))
		theoretical_distributions[label] = get_theoretical_hidden_out(input_vector, m_weights, s_weights)

	# theoretical = label
	# actual = other_label
	for label in range(10):
		for other_label in range(10):
			print('({}, {})'.format(label, other_label))
			print('{}'.format(kl_divergence(theoretical_distributions[label][0],
											theoretical_distributions[label][1],
											fits[other_label][0], fits[other_label][1])))

def plot_mu_sigma(samples, weights, fits):
	color_dict = {0:'xkcd:black',
				1: 'xkcd:aqua',
				2: 'xkcd:blue',
				3: 'xkcd:brown',
				4: 'xkcd:coral',
				5: 'xkcd:gold',
				6: 'xkcd:green',
				7: 'xkcd:lavender',
				8: 'xkcd:orange',
				9: 'xkcd:pink'
				}

	m_weights, s_weights = norm.fit(weights)
	# Actual output plotting
	for sample_class in samples:
		label = sample_class[0][1]
		color = color_dict[label]
		xs = [point[0] for point in fits[label]]
		ys = [point[1] for point in fits[label]]
		plt.scatter(xs, ys, c = color)

	handles = [mpatches.Patch(color = color_dict[label], label = label) for label in color_dict]
	plt.title('Actual Output Fits')
	plt.xlabel('\u03BC')
	plt.ylabel('\u03C3')
	plt.legend(handles = handles)
	plt.show()


	# Theoretical vs Averaged Actual
	plt.clf()
	fits = average_fits(fits)

	for sample_class in samples:
		label = sample_class[0][1]
		color = color_dict[label]
		input_vectors = [x[0] for x in sample_class]
		input_vector = torch.div(sum(input_vectors), len(input_vectors))
		m_theoretical, s_theoretical = get_theoretical_hidden_out(input_vector, m_weights, s_weights)
		plt.scatter(m_theoretical, s_theoretical, c = color)
		plt.scatter(fits[label][0], fits[label][1], c = color, alpha = 0.25)
	plt.title('Theoretical vs Actual Output Fits')
	plt.xlabel('\u03BC')
	plt.ylabel('\u03C3')
	plt.legend(handles = handles)
	plt.show()



# Experiments
mnist_train_data, mnist_test_data = get_mnist_data()

net = Simple_MNIST_Model()

net.load_state_dict(torch.load('trained_mnist.pt'))
net.eval()

samples = get_correct_samples(net, mnist_test_data, 3)

weights = get_model_weights(net)

remove_softmax_layer2(net)

net_outs = sample_test_outputs(net, samples)

fits = get_fits(net_outs)

plot_mu_sigma(samples, weights, fits)


