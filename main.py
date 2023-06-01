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

import database as DBFunc
from task1 import Task1
import task2
import task2_1
import task3
import task4
import task5
import task6
import task8
def main():
	[caltech101, data_loader] = importCaltech.importData()
	print('\n')

	menu = {}
	menu['1']="Save CSV even data"
	menu['2']="make a query"
	menu['21']="save distance on file"
	menu['3']="make a query on label"
	menu['4']="Associating Label"
	menu['5']= "PageRank"
	menu['6']="Features Latenti"
	menu['8']="Features Latenti label-Label"
	menu['13']="Exit"
	menu['14']="Test"
	while True:
		options=menu.keys()
		for entry in options:
			print(entry, menu[entry])

		selection=input("Please Select:")
		if selection =='1':
			task1 = Task1()
			task1.processing(caltech101,ReteNeurale)
		elif selection == '2':
			task2.start(caltech101, ReteNeurale)
		elif selection == '21':
			task2_1.processing(caltech101, ReteNeurale)
		elif selection == '3':
			task3.start(caltech101, ReteNeurale)
		elif selection == '4':
			task4.processing(caltech101, ReteNeurale)
		elif selection == '5':
			task5.processing(caltech101, ReteNeurale)
		elif selection == '6':
			task6.processing(caltech101, ReteNeurale)
		elif selection == '8':
			task8.processing(caltech101, ReteNeurale)
		elif selection == '13':
			break
		elif selection == '14':
			DB= DBFunc.getDB(3)
			origin = np.array([[0, 0, 0], [0, 0, 0]])  # origin point

			plt.quiver(DB[0],DB[1],DB[2])
			plt.show()
		else:
			print("Unknown Option Selected!")


if __name__ == "__main__":
	main()








