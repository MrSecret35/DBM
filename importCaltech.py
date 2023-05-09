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

print("PyTorch Version: ", torch.__version__)
print("Torchvision Version: ", torchvision.__version__)




def importData():
    print("import data from Caltech101")
    ROOT = ""
    Caltech101 = torchvision.datasets.Caltech101(root=ROOT, download='true')
    data_loader = torch.utils.data.DataLoader(Caltech101,
                                              batch_size=4,
                                              shuffle=True,
                                              num_workers=8)
    print("save data in to file Caltech101.json")
    return [Caltech101,data_loader]
