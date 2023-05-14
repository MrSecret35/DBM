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
import task2

def main():
    [caltech101, data_loader] = importCaltech.importData()
    print('\n')
    
    menu = {}
	menu['1']="Save CSV even data"
	menu['2']="make a range query (range = 4)"
	menu['8']="Exit"
	menu['9']="Test"
	while True:
	    options=menu.keys()
	    for entry in options:
	        print(entry, menu[entry])
	
	    selection=input("Please Select:")
	    if selection =='1':
	        task1 = Task1()
	    	task1.processing(caltech101,ReteNeurale)
	    elif selection == '2':
	        task2.start(caltech101, ReteNeurale, features)
	    elif selection == '8':
	        break
	    elif selection == '9':
	        [vector_last, vector_avgpool, vector_layer3, class_id] = task1.Extractor(0, caltech101, ReteNeurale, features)
	        print(vector_avgpool)
	    else:
	      print("Unknown Option Selected!")


if __name__ == "__main__":
    main()








