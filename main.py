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
from torchvision.io import read_image, ImageReadMode
from torchvision.models import resnet50, ResNet50_Weights

from NeuralNetwork import ReteNeurale, features
import genericFunction as GF
import importCaltech


from task1 import Task1

def main():
    [caltech101, data_loader] = importCaltech.importData()
    task1 = Task1()
    task1.processing(caltech101,ReteNeurale)

if __name__ == "__main__":
    main()



