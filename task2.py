import torch
import torch.nn as nn
import torchvision
import csv
import numpy
import matplotlib.pyplot as plt
import math

import task1


def start(Caltech101, ReteNeurale, features):
    fileVectorLast = open('Data\dataVectorLast.csv', 'r')
    readerVectorLast = csv.reader(fileVectorLast, delimiter=';')
    res = numpy.array(list(readerVectorLast))

    [vector_last, vector_avgpool, vector_layer3, class_id] = task1.Extractor(3, Caltech101, ReteNeurale, features)

    img,label = Caltech101[3]
    plt.imshow(img)
    plt.show()
    listaSim = []
    for i in range(len(res)):
        d= distanza(vector_last.detach().numpy(), res[i])
        listaSim.append((i*2,d))

    #print(listaSim)
    #print(listaSim[0][1])
    min= listaSim[0]
    for i in range(len(res)):
        if listaSim[i][1] < min[1]:
            min = listaSim[i]

    print(min[0])
    imgRes, labelRes = Caltech101[min[0]]
    plt.imshow(imgRes)
    plt.show()
def distanza(v1, v2):
    if len(v1) != len(v2) :
        print("error")
        return -1

    res = 0
    for i in range(len(v1)):
        res += (v1[i].item()-float(v2[i].item()))**2

    return math.sqrt(res)

