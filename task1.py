import torch
import torch.nn as nn
import torchvision
import csv
from NeuralNetwork import *
import genericFunction as GF
from tqdm import tqdm

#from tensorly.decomposition import parafac
#dec = parafac(vector_last.unsqueeze(-1).detach().numpy(),rank=3)

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
		fileVectorLast = open('Data\dataVectorLast.csv', 'w', newline='')
	    fileVector_avgpool = open('Data\dataVectorAVGPool.csv', 'w', newline='')
	    fileVector_layer3= open('Data\dataVectorLayer3.csv', 'w', newline='')
	
	    writerVL = csv.writer(fileVectorLast, delimiter=';')
	    writerVA = csv.writer(fileVector_avgpool, delimiter=';')
	    writerV3 = csv.writer(fileVector_layer3, delimiter=';')
	
	    print("Start csv data Creation")
	    for i in range(Caltech101.__len__()):
	        img, label = Caltech101[i]
	        if i % 2 == 0 and torchvision.transforms.functional.get_image_num_channels(img) != 1:
	        	tensor = GF.IMGtoTensor(img)
				
				vector_last=self.getFC(tensor,model)
				vector_avgpool=self.getAvgpoolFlatten('avgpool')
				vector_layer3=self.getLayer3Flatten('layer3')
				
	            writerVL.writerow(vector_last.detach().numpy())
	            writerVA.writerow(vector_avgpool.detach().numpy())
	            writerV3.writerow(vector_layer3.detach().numpy())
	
	        #TODO: cancellare questo if, è solo per vedere che non si blocchi
	        if i % 500 == 0:
	            print("fatti:")
	            print(i)
	
	    print("Finish csv data Creation")
