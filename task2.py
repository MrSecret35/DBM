import csv
import numpy
import matplotlib.pyplot as plt
import genericFunction as GF
import math
import torch

import database as DBFunc
from tqdm import tqdm
from task1 import Task1


def processing(dataset, ReteNeurale):
    task1 = Task1()

    # take ID of IMG for the query
    ID_img_query= GF.getIDImg(dataset)

    # take number of images to print
    n = int(input("insert n (numero immagini da stampare):"))

    # take DB: matrix corresponding to the csv file indicated by the ID_space
    ID_space = DBFunc.IDSpace()
    DB = DBFunc.getDB(ID_space)
    id_row = DBFunc.getDBID()

    # take queryVector: vector taken by NeuralNetwork by ID_img and ID_space
    QueryVector = task1.getVectorbyID(ID_img_query, dataset, ReteNeurale, ID_space)




    #get listaSim: list containing tuples (2 elements) corresponding to the ID of the img and its similarity(distance) with the query img
    listaSim = getSimilarityVector(QueryVector,DB)
    listaSim = sorted(listaSim, key=lambda tup: tup[1])

    listaSim = [ DBFunc.getIDfromRow(i,id_row) for (i,j) in listaSim[0:n]]
    GF.printNImageCompare(ID_img_query,listaSim,dataset)

# distance(v1, v2)
# v1: vector 1
# v2: vector 2
#
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
#
# return a list containing tuples (2 elements)
#   corresponding to the ID of the img and its similarity with the query img
def getSimilarityVector(query, dataset):
    listaSim = []
    for i in range(len(dataset)):
        d = distance(query, dataset[i])
        listaSim.append((i+1,d))
    return listaSim
