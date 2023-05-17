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

    #ask what image
    ID_img_query=-1
    while ID_img_query==-1 or ID_img_query >= len(Caltech101) :
        ID_img_query = int(input("insert an ID for an Image (odd number)(max " + str(len(Caltech101)-1) + "):  "))
        if ID_img_query >= len(Caltech101):
            print("insert a valid selection (max " + str(len(Caltech101)-1) + "):  ")

    # ask N image to print
    n = int(input("insert n (numero immagini):"))

    # ask ID spase
    ID_space = ''
    while ID_space != '1' and ID_space != '2' and ID_space != '3':
        ID_space = input("select space: \n 1 - layer3\n 2 - avg\n 3 - last\n")
        if(ID_space != '1' and ID_space != '2' and ID_space != '3'):
            print("insert a valid selection")

    # take queryVector: vector taken by NeuralNetwork by ID_img and ID_space
    QueryVector = task1.getVectorbyID(ID_img_query, Caltech101, ReteNeurale, ID_space)

    # take DB: matrix corresponding to the csv file indicated by the ID_space
    DB = getDB(ID_space)

    #get listaSim: list containing tuples (2 elements) corresponding to the ID of the img and its similarity with the query img
    listaSim = getSimilarityVector(QueryVector,DB)
    listaSim=sorted(listaSim, key=lambda tup: tup[1])

    img_query, label_query = Caltech101[ID_img_query]
    printNImage(img_query, listaSim, n, Caltech101)


# distance(v1, v2)
# v1: vector 1
# v2: vector 2
# calculates and returns the Euclidean distance between 2 vectors
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

def printNImage(img, lista, n, dataset):
    f, axarr = plt.subplots(2, n)
    axarr[0][0].imshow(img)
    axarr[0][0].set_title("immagine scelta")
    for i in range(n):
        imgRes, labelRes = dataset[lista[i][0]]
        axarr[1][i].imshow(imgRes)
        axarr[1][i].set_title("img num: " + str(i+1))
    plt.show()

def getDB(ID_space):
    fileVector = {}
    if ID_space == '1':
        fileVector = open('Data\dataVectorLayer3.csv', 'r')
    elif ID_space == '2':
        fileVector = open('Data\dataVectorAVGPool.csv', 'r')
    elif ID_space == '3':
        fileVector = open('Data\dataVectorLast.csv', 'r')

    readerVector = csv.reader(fileVector, delimiter=';', quoting=csv.QUOTE_NONNUMERIC)
    DB = numpy.array(list(readerVector))

    return DB