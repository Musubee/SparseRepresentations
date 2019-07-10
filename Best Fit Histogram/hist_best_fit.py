import torchvision.models as models

import pandas as pd

import numpy as numpy
import matplotlib.pyplot as plt

vgg = models.vgg16(pretrained = True)
vgg_weights = extract_torch_weights(vgg)

df = vgg_weights

