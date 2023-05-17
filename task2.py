import csv
import numpy
import matplotlib.pyplot as plt
import genericFunction as GF
import math
import torch
from tqdm import tqdm


from task1 import Task1


def start(Caltech101, ReteNeurale):
    task1 = Task1()

    #TODO: fare i controlli per gli ID
    #ask what image
    ID_img = int(input("insert an ID for an Image (odd number):"))

    n = int(input("insert n (numero immagini):"))

    ID_space = ''
    while ID_space != '1' and ID_space != '2' and ID_space != '3':
        ID_space = input("select space: \n 1 - layer3\n 2 - avg\n 3 - last\n")
        if(ID_space != '1' and ID_space != '2' and ID_space != '3'):
            print("insert a valid selection")

    fileVector ={}
    if ID_space == '1':
        fileVector = open('Data\dataVectorLayer3.csv', 'r')
    elif ID_space == '2':
        fileVector = open('Data\dataVectorAVGPool.csv', 'r')
    elif ID_space == '3':
        fileVector = open('Data\dataVectorLast.csv', 'r')


    readerVector = csv.reader(fileVector,delimiter=';',quoting=csv.QUOTE_NONNUMERIC)
    DB = numpy.array(list(readerVector))

    img, label = Caltech101[ID_img]


    [vector_last, vector_avgpool, vector_layer3] = task1.Extractor(ID_img, Caltech101, ReteNeurale)
    QueryVector = []
    if ID_space == '1':
        QueryVector = vector_layer3.detach().numpy()
    elif ID_space == '2':
        QueryVector = vector_avgpool.detach().numpy()
    elif ID_space == '3':
        QueryVector = vector_last.detach().numpy()

    listaSim = getSimilarityVector(QueryVector,DB)


    #TODO: trovare gli n simili (non solo 1)
    #min= listaSim[0]
    #for i in range(len(DB)):
    #    if listaSim[i][1] < min[1]:
    #        min = listaSim[i]

    listaSim=sorted(listaSim, key=lambda tup: tup[1])

    f, axarr = plt.subplots(2, n)
    axarr[0][0].imshow(img)
    axarr[0][0].set_title("immagine scelta")
    for i in range(n):
        imgRes, labelRes = Caltech101[listaSim[i][0]]
        axarr[1][i].imshow(imgRes)
        axarr[1][i].set_title("img num: " + str(i))
    plt.show()

def distance(v1, v2):
    if len(v1) != len(v2):
        print("error")
        return -1

    v1 = [float(v) for v in v1]
    v2 = [float(v) for v in v2]

    v1 = torch.tensor([v1])
    v2 = torch.tensor([v2])

    return torch.cdist(v1,v2,1)

def getSimilarityVector(query, dataset):
    listaSim = []
    for i in tqdm(range(len(dataset))):
        d = distance(query, dataset[i])
        listaSim.append((i*2,d))
    return listaSim