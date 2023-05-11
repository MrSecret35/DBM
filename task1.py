import torch
import torch.nn as nn
import torchvision
import csv
from NeuralNetwork import *
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
        #lista di tuple di tensori (id,fc_tensor)
        fc_list = list()
        #lista di tuple di tensori, normale e flattened (id, normale, flattened)
        avgpool_list = list()
        #lista di tuple di tensori (id, layer3_tensor)
        layer3_list = list()

        #for i in tqdm(range(dataset.__len__())):
        for i in tqdm(range(2)):
            img, label = dataset[i]
            if i % 2 == 0 and torchvision.transforms.functional.get_image_num_channels(img) != 1:
                tensor = GF.IMGtoTensor(img)
                fc_list.append((i,self.getFC(tensor,model)))
                tuple = (i,self.getHook('avgpool'),self.getAvgpoolFlatten('avgpool'))
                avgpool_list.append(tuple)
                layer3_list.append((i,self.getLayer3Flatten('layer3')))