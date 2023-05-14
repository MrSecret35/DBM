import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
from pathlib import Path
import os
import copy
import json
from torchvision.io import read_image, ImageReadMode
from torchvision.models import resnet50, ResNet50_Weights

# prepare the function to preprocess images to be compatible with ResNet50
default_weights = ResNet50_Weights.DEFAULT
preprocess = default_weights.transforms()

def get_features(name,features):
    def hook(model, input, output):
        features[name] = output.detach()
    return hook

def IMGtoTensor(img):
  if torchvision.transforms.functional.get_image_num_channels(img) != 1:
    proc_img = preprocess(img).unsqueeze(0)
    return proc_img
  else:
    print ("incompatiable image format -- try another one")