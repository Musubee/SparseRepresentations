import torch
from torchvision import models


def threshold_tensor(tensor, t):
	'''
	Thresholds values in tensor whose magnitude is less than t to 0.
	
	Key Parameters:
	tensor (Torch.Tensor) = tensor to be thresholded
	t (float) = threshold value
	'''
	if len(tensor.size()) == 0:
		return
	elif len(tensor.size()) == 1:
		for i in range(tensor.size()[0]):
			if abs(tensor[i]) <= t:
				tensor[i] = 0
	else:
		for i in range(tensor.size()[0]):
			threshold_tensor(tensor[i],t)

def save_alexnet_thresholded_models():
	'''Saves thresholded AlexNet models (t = 1, 0.1, 0.01, 0.001, 0) to models/Alexnet/'''
	base_path = 'models/AlexNet/alexnet'
	alexnet = models.alexnet(pretrained = True)

	for t in [1, 0.1, 0.01, 0.001, 0]:
		for key in alexnet.state_dict().keys():
			if 'weight' in key:
				print('AlexNet: ' + key)
				threshold_tensor(alexnet.state_dict()[key], t)

		torch.save(alexnet, base_path + '_t' + str(t) + '.pt')

		# Refresh model to original state
		alexnet = models.alexnet(pretrained = True)

def save_vgg16_thresholded_models():
	'''Saves thresholded VGG16 models (t = 1, 0.1, 0.01, 0.001, 0) to models/VGG16/'''
	base_path = 'models/VGG16/vgg16_'
	vgg16 = models.vgg16(pretrained = True)

	for t in [1, 0.1, 0.01, 0.001, 0]:
		for key in vgg16.state_dict().keys():
			if 'weight' in key:
				print('VGG16: ' + key)
				threshold_tensor(vgg16.state_dict()[key], t)

		torch.save(vgg16, base_path + 't' + str(t) + '.pt')

		vgg16 = models.vgg16(pretrained = True)

def save_googlenet_thresholded_models():
	'''Saves thresholded GoogLeNet models (t = 1, 0.1, 0.01, 0.001, 0) to models/GoogLeNet/'''
	base_path = 'models/GoogLeNet/googlenet_'
	googlenet = models.googlenet(pretrained = True)

	for t in [1, 0.1, 0.01, 0.001, 0]:
		for key in googlenet.state_dict().keys():
			if 'weight' in key:
				print('GoogLeNet: ' + key)
				threshold_tensor(googlenet.state_dict()[key], t)

		torch.save(googlenet, base_path + 't' + str(t) + '.pt')

		googlenet = models.googlenet(pretrained = True)

def save_resnet18_thresholded_models():
	'''Saves thresholded ResNet18 models (t = 1, 0.1, 0.01, 0.001, 0) to models/ResNet18/'''
	base_path = 'models/ResNet18/resnet18_'
	resnet18 = models.resnet18(pretrained = True)

	for t in [1, 0.1, 0.01, 0.001, 0]:
		for key in resnet18.state_dict().keys():
			if 'weight' in key:
				print('ResNet18: ' + key)
				threshold_tensor(resnet18.state_dict()[key], t)

		torch.save(resnet18, base_path + 't' + str(t) + '.pt')

		resnet18 = models.resnet18(pretrained = True)


save_googlenet_thresholded_models()
save_resnet18_thresholded_models()
save_alexnet_thresholded_models()
save_vgg16_thresholded_models()





