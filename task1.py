import torch
import torch.nn as nn
import torchvision
import csv

import genericFunction as GF


def createData(Caltech101, ReteNeurale, features):
    d = Extractor(1, Caltech101, ReteNeurale, features)
    vector_last_list = []
    vector_avgpool_list = []
    vector_layer3_list = []
    count = 500
    #for i in range(Caltech101.__len__()):
    for i in range(2):
        img, label = Caltech101[i]
        if i % 2 == 0 and torchvision.transforms.functional.get_image_num_channels(img) != 1:
            [vector_last, vector_avgpool, vector_layer3, class_id] = Extractor(i, Caltech101, ReteNeurale, features)
            print(vector_last.item())
    torch.save(vector_last_list, 'data/vector_last_list.pt')
    torch.save(vector_last_list, 'data/vector_avgpool_list.pt')
    torch.save(vector_last_list, 'data/vector_layer3_list.pt')

def Extractor(imgID, Caltech101, ReteNeurale, features):
    img, label = Caltech101[imgID]
    tensImg = GF.IMGtoTensor(img)
    prediction = ReteNeurale(tensImg).squeeze(0).softmax(0)
    class_id = prediction.argmax().item()

    vector_last = ReteNeurale(tensImg).squeeze(0)

    temp_tensor_avgpool = torch.tensor(features['avgpool'].cpu().numpy().squeeze(0))
    tensor_avgpool = temp_tensor_avgpool.reshape(1, 1, -1)
    k = nn.AvgPool1d(2, stride=2)
    vector_avgpool = k(tensor_avgpool).flatten()

    tensor_layer3 = torch.tensor(features['layer3'].cpu().numpy().squeeze(0))
    m = nn.AvgPool2d(14, stride=1)
    vector_layer3 = m(tensor_layer3).flatten()

    return [vector_last, vector_avgpool, vector_layer3, class_id]

def ExtractorVectorLast(imgID, Caltech101, ReteNeurale, features):
    img, label = Caltech101[imgID]
    tensImg = GF.IMGtoTensor(img)

    vector_last = ReteNeurale(tensImg).squeeze(0)

    return vector_last

def ExtractorVectorAVGPool(imgID, Caltech101, ReteNeurale, features):
    img, label = Caltech101[imgID]
    tensImg = GF.IMGtoTensor(img)

    vector_last = ReteNeurale(tensImg).squeeze(0)

    temp_tensor_avgpool = torch.tensor(features['avgpool'].cpu().numpy().squeeze(0))
    tensor_avgpool = temp_tensor_avgpool.reshape(1, 1, -1)
    k = nn.AvgPool1d(2, stride=2)
    vector_avgpool = k(tensor_avgpool).flatten()

    return vector_avgpool

def ExtractorVectorLayer3(imgID, Caltech101, ReteNeurale, features):
    img, label = Caltech101[imgID]
    tensImg = GF.IMGtoTensor(img)

    vector_last = ReteNeurale(tensImg).squeeze(0)

    tensor_layer3 = torch.tensor(features['layer3'].cpu().numpy().squeeze(0))
    m = nn.AvgPool2d(14, stride=1)
    vector_layer3 = m(tensor_layer3).flatten()

    return vector_layer3
