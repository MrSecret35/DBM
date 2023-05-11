import torch
import torch.nn as nn
import torchvision
import csv
import tensorly as tl
from tensorly.decomposition import parafac

import genericFunction as GF


def createData2(Caltech101, ReteNeurale, features):
    [vector_last, vector_avgpool, vector_layer3, class_id] = Extractor(0, Caltech101, ReteNeurale, features)
    vector=vector_last.detach().numpy()
    #numpy.savetxt('data.csv', vector, delimiter=",")

    print(vector)
    print(len(vector))

    f = open('data.csv', 'w')
    writer = csv.writer(f)
    writer.writerow(vector)

    dec = parafac(vector_last.unsqueeze(-1).detach().numpy(),rank=3)



def createData(Caltech101, ReteNeurale, features):
    fileVectorLast = open('Data\dataVectorLast.csv', 'w', newline='')
    fileVector_avgpool = open('Data\dataVectoAVGPool.csv', 'w', newline='')
    fileVector_layer3= open('Data\dataVectorLayer3.csv', 'w', newline='')

    writerVL = csv.writer(fileVectorLast, delimiter=';')
    writerVA = csv.writer(fileVector_avgpool, delimiter=';')
    writerV3 = csv.writer(fileVector_layer3, delimiter=';')

    print("Start csv data Creation")
    for i in range(Caltech101.__len__()):
        img, label = Caltech101[i]
        if i % 2 == 0 and torchvision.transforms.functional.get_image_num_channels(img) != 1:
            [vector_last, vector_avgpool, vector_layer3, class_id] = Extractor(i, Caltech101, ReteNeurale, features)

            writerVL.writerow(vector_last.detach().numpy())
            writerVA.writerow(vector_avgpool.detach().numpy())
            writerV3.writerow(vector_layer3.detach().numpy())

        #TODO: cancellare questo if, è solo per vedere che non si blocchi
        if i % 500 == 0:
            print("fatti:")
            print(i)

    print("Finish csv data Creation")

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
