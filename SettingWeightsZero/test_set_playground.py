from gluoncv.data import ImageNet
from mxnet.gluon.data import DataLoader
from mxnet.gluon.data.vision import transforms

test_trans = transforms.Compose([
	transforms.RandomResizedCrop(224),
	transforms.ToTensor()
])

test_data = DataLoader(
	ImageNet(root = '.', train = False).transform_first(test_trans),
	batch_size = 128, shuffle = True)