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

import importCaltech

[Caltech101,data_loader] = importCaltech.importData()




