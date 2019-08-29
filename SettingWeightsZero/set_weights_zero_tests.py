import unittest
import torch

import set_weights_zero as swz

class ThresholdTensorTests(unittest.TestCase):
	'''
	Testing Strategy

	Partition the input as follows: 
	len(tensor.size()) = 0, 1, >1
	len(tensor.size()[i]) = 1, >1
	t = 0, 0.001, 0.01, 0.1, 1 (threshold values to be used in actual experiment)

	Covers each part of partitioning at least once. 

	Have to individually check values b/c bool values are ambiguous for tensors w/ different values.
	'''

	def setUp(self):
		pass

	# Returns True if empty Tensor is returned
	# 
	# covers: len(tensor.size()) = 0, t = 0
	def test_empty(self):
		tensor = torch.tensor([])
		swz.threshold_tensor(tensor, 0)
		self.assertEqual(len(tensor.size()), len(torch.tensor([]).size()))

	# Returns True if original Tensor is returned
	#
	# covers: len(tensor.size()) = 1; len(tensor.size()[i]) = 1, t = 0
	def test_single_dimension_no_thresholding(self):
		tensor = torch.tensor([1, 2, 3])
		swz.threshold_tensor(tensor, 0)
		correct_tensor = torch.tensor([1, 2, 3])
		for i in range(correct_tensor.size()[0]):
			self.assertEqual(tensor[i], correct_tensor[i])

	# Returns True if original Tensor = [0, 1] after modification
	#
	# covers: len(tensor.size()) = 1; len(tensor.size()[i]) > 1; t = 0.1
	def test_single_dimension_some_thresholding(self):
		tensor = torch.tensor([0.01, 1], dtype = torch.float)
		swz.threshold_tensor(tensor, 0.1)
		correct_tensor = torch.tensor([0, 1], dtype = torch.float)
		for i in range(correct_tensor.size()[0]):
			self.assertEqual(tensor[i], correct_tensor[i])

	# Returns True if zero Tensor of appropriate dimensions is returned
	#
	# covers: len(tensor.size()) > 1; len(tensor.size()[i]) =,> 1; t = 1
	def test_multi_dimension_all_thresholded(self):
		tensor = torch.tensor([[[0.01, 0.1], [0.1, 0.01]], [[0.001, 0.2], [0.3, 0.4]]], dtype = torch.float)
		swz.threshold_tensor(tensor, 1)
		correct_tensor = torch.tensor([[[0, 0], [0, 0]], [[0, 0], [0, 0]]], dtype = torch.float)
		for i in range(correct_tensor.size()[0]):
			for j in range(correct_tensor.size()[1]):
				for k in range(correct_tensor.size()[2]):
					self.assertEqual(tensor[i][j][k], correct_tensor[i][j][k])

	# Returns True if returned Tensor = [[1, 0], [0, 0.01]] after modification
	#
	# covers: len(tensor.size()) > 1; len(tensor.size()[i]) =,> 1; t = 0.001
	def test_t_1eneg3(self):
		tensor = torch.tensor([[1, 0.0001], [0.0004, 0.01]], dtype = torch.float)
		swz.threshold_tensor(tensor, 0.001)
		correct_tensor = torch.tensor([[1, 0], [0, 0.01]], dtype = torch.float)
		for i in range(correct_tensor.size()[0]):
			for j in range(correct_tensor.size()[1]):
				self.assertEqual(tensor[i][j], correct_tensor[i][j])

	# Returns True if returned Tensor = [[0], [0.013]] after modification
	#
	# covers: len(tensor.size()) > 1; len(tensor.size()[i]) =,> 1; t = 0.01
	def test_t_1eneg2(self):
		tensor = torch.tensor([[0.0099], [0.013]], dtype = torch.float)
		swz.threshold_tensor(tensor, 0.01)
		correct_tensor = torch.tensor([[0], [0.013]], dtype = torch.float)
		for i in range(correct_tensor.size()[0]):
			for j in range(correct_tensor.size()[1]):
				self.assertEqual(tensor[i][j], correct_tensor[i][j])

	# Returns True if returned Tensor = [[0.2, 0.3, 0], [0, 0.6, 2]] after modification
	#
	# covers: len(tensor.size()) > 1; len(tensor.size()[i]) -,> 1; t = 0.1
	def test_t_1eneg1(self):
		tensor = torch.tensor([[0.2, 0.3, 0.04], [0.05, 0.6, 2]], dtype = torch.float)
		swz.threshold_tensor(tensor, 0.1)
		correct_tensor = torch.tensor([[0.2, 0.3, 0], [0, 0.6, 2]], dtype = torch.float)
		for i in range(correct_tensor.size()[0]):
			for j in range(correct_tensor.size()[1]):
				self.assertEqual(tensor[i][j], correct_tensor[i][j])


if __name__ == '__main__':
	unittest.main()