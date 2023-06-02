import csv
import numpy
import matplotlib.pyplot as plt
import genericFunction as GF
import math
import torch
import database as DBFunc
from tqdm import tqdm

from task1 import Task1


def start(Caltech101, ReteNeurale):
    task1 = Task1()

    #ask what image
    ID_img_query= GF.getIDImg(Caltech101)
    id_row = DBFunc.getDBID()

    # ask N image to print
    n = int(input("insert n (numero immagini):"))

    # ask ID spase
    ID_space = DBFunc.IDSpace()

    # take queryVector: vector taken by NeuralNetwork by ID_img and ID_space
    QueryVector = task1.getVectorbyID(ID_img_query, Caltech101, ReteNeurale, ID_space)

    # take DB: matrix corresponding to the csv file indicated by the ID_space
    DB = DBFunc.getDB(ID_space)

    #get listaSim: list containing tuples (2 elements) corresponding to the ID of the img and its similarity with the query img
    listaSim = getSimilarityVector(QueryVector,DB)
    listaSim=sorted(listaSim, key=lambda tup: tup[1])

    img_query, label_query = Caltech101[ID_img_query]
    printNImage(img_query, listaSim, n, Caltech101,id_row)


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

# getSimilarityVector(query, dataset)
# query:    vector from which to calculate the similarity
# dataset   list of vector (same size query)
# return a list containing tuples (2 elements)
#   corresponding to the ID of the img and its similarity with the query img
def getSimilarityVector(query, dataset):
    listaSim = []
    for i in range(len(dataset)):
        d = distance(query, dataset[i])
        listaSim.append((i+1,d))
    return listaSim

# printNImage(img, lista, n, dataset)
# img:      image to be printed in the first row
# list:     list from which to get the ID of img for the second row
# n:        number of img to print in the second line
# dataset:  dataset to get the img from id
# prints 2 lines, the first containing img and the second containing the first n img of the list
def printNImage(img, list, n, dataset,id_row):
    f, axarr = plt.subplots(2, n)
    axarr[0][0].imshow(img)
    axarr[0][0].set_title("immagine scelta")
    for i in range(n):
        imgRes, labelRes = dataset[ DBFunc.getIDfromRow(list[i][0],id_row)]
        axarr[1][i].imshow(imgRes)
        axarr[1][i].set_title("img num: " + str(i+1))
    plt.show()