import torch
import torch.nn as nn
import torchvision
import csv

from NeuralNetwork import *
import NeuralNetwork as NN
import genericFunction as GF
from tqdm import tqdm

class Task1():
	def __init__(self) -> None:
		pass

	def getFC(self,tensor, ReteNeurale):
		return ReteNeurale(tensor).squeeze(0)

	def getHook(self,tensorName):
		return features[tensorName].squeeze(0)

	def getAvgpoolFlatten(self,tensorName):
		output = self.getHook(tensorName).reshape(1,1,-1)
		k = nn.AvgPool1d(2,stride=2)
		return k(output).flatten()

	def getLayer3Flatten(self,tensorName):
		m = nn.AvgPool2d(14,stride=1)
		return m(self.getHook(tensorName)).flatten()

	def processing(self,dataset,model):
		
		fileIDtoRow = open('Data\IDtoRow.csv', 'w', newline='')
		fileVectorLast = open('Data\dataVectorLast.csv', 'w', newline='')
		fileVector_avgpool = open('Data\dataVectorAVGPool.csv', 'w', newline='')
		fileVector_layer3= open('Data\dataVectorLayer3.csv', 'w', newline='')

		writerID = csv.writer(fileIDtoRow, delimiter=';')
		writerVL = csv.writer(fileVectorLast, delimiter=';')
		writerVA = csv.writer(fileVector_avgpool, delimiter=';')
		writerV3 = csv.writer(fileVector_layer3, delimiter=';')

		print("Start csv data Creation")
		row=1
		for i in tqdm(range(dataset.__len__())):
			img, label = dataset[i]
			if i % 2 == 0 and torchvision.transforms.functional.get_image_num_channels(img) != 1:
				tensor = NN.IMGtoTensor(img)

				vector_last=self.getFC(tensor,model)
				vector_avgpool=self.getAvgpoolFlatten('avgpool')
				vector_layer3=self.getLayer3Flatten('layer3')

				writerID.writerow([i,row])
				writerVL.writerow(vector_last.detach().numpy())
				writerVA.writerow(vector_avgpool.detach().numpy())
				writerV3.writerow(vector_layer3.detach().numpy())
				row+=1
		print("Finish csv data Creation")

	# Extractor(imgID, dataset, model)
	# imgID: ID dell'immagine di cui estrarre le features
	# dataset : dataset dell'img
	# model: rete neurale
	def Extractor(self, imgID, dataset, model):
		img, label = dataset[imgID]
		tensor = NN.IMGtoTensor(img)

		#prediction = model(tensor).squeeze(0).softmax(0)
		#class_id = prediction.argmax().item()

		vector_last = self.getFC(tensor, model)
		vector_avgpool = self.getAvgpoolFlatten('avgpool')
		vector_layer3 = self.getLayer3Flatten('layer3')


		return [vector_last, vector_avgpool, vector_layer3]

	def getVectorbyID(self, imgID, dataset, model, ID_space):
		img, label = dataset[imgID]
		tensor = NN.IMGtoTensor(img)
		resVector = []

		vector_last = self.getFC(tensor, model)
		vector_avgpool = self.getAvgpoolFlatten('avgpool')
		vector_layer3 = self.getLayer3Flatten('layer3')

		if ID_space == 1:
			resVector = vector_layer3.detach().numpy()
		elif ID_space == 2:
			resVector = vector_avgpool.detach().numpy()
		elif ID_space == 3:
			resVector = vector_last.detach().numpy()

		return resVector