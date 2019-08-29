import torch
from torchvision import models
from torchvision import datasets
from torchvision import transforms

from PIL import Image

import matplotlib.pyplot as plt

resnet18 = models.resnet18(pretrained = True)
resnet18_t0 = torch.load('models/ResNet18/resnet18_t_0.pt')
resnet18.eval()
resnet18_t0.eval()


# 1000 folders for each image class
# Problem: testing networks on this data
# Thoughts: 
# Able to extract pillow image and label
# Next apply transform; actually check out on one image to see if transforms are necessary

# from: https://www.learnopencv.com/pytorch-for-beginners-image-classification-using-pre-trained-models/
transform1 = transforms.Compose([
	transforms.Resize(256),
	transforms.CenterCrop(224),
	transforms.ToTensor(),
	transforms.Normalize(
	mean = [0.485, 0.456, 0.406],
	std = [0.229, 0.224, 0.225]
	)
	])
transform2 = transforms.ToTensor()

dataset = datasets.ImageFolder(root = 'imagenetv2-matched-frequency')

def test_resnet(orig_resnet, t_resnet, dataset):
	'''Returns top-1 and top-5 error on test set (ImageNetV2)'''
	orig_top1_correct = 0
	orig_top5_correct = 0
	t_top1_correct = 0
	t_top5_correct = 0
	total = 10000

	for i, data in enumerate(dataset):
		print(str(i))
		img, label = data

		#transform image and prepare a batch to feed 
		img_t = transform2(img)
		batch_t = torch.unsqueeze(img_t, 0)

		orig_out = orig_resnet(batch_t)

		_, orig_indices = torch.sort(orig_out, descending = True)
		if label in orig_indices[0][:5]:
			if label == orig_indices[0][0]:
				orig_top1_correct += 1
			orig_top5_correct += 1

		t_out = t_resnet(batch_t)

		_, t_indices = torch.sort(t_out, descending = True)
		if label in t_indices[0][:5]:
			if label == t_indices[0][0]:
				t_top1_correct += 1
			t_top5_correct += 1

		print('Original running accuracy')
		print('Top 1 Acc: ' + str(orig_top1_correct/(i+1)) + '; Top 5 Acc: ' + str(orig_top5_correct/(i+1)))
		print('Thresholded running accuracy:')
		print('Top 1 Acc: ' + str(t_top1_correct/(i+1)) + '; Top 5 Acc: ' + str(t_top5_correct/(i+1)))
	



test_resnet(resnet18, resnet18_t0, dataset)

