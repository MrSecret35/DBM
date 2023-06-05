import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
from pathlib import Path
import os
import copy
import json
import csv
from torchvision.io import read_image, ImageReadMode
from torchvision.models import resnet50, ResNet50_Weights



def getIDImg(Caltech101):
    ID_img_query = -1
    while ID_img_query == -1 or ID_img_query >= len(Caltech101):
        ID_img_query = int(input("insert an ID for an Image (odd number)(max " + str(len(Caltech101) - 1) + "):  "))
        if ID_img_query >= len(Caltech101):
            print("insert a valid selection (max " + str(len(Caltech101) - 1) + "):  ")
    return ID_img_query

def printIMG(IDImg,text,dataset):
    img, label = dataset[IDImg]
    plt.imshow(img)
    plt.title("IMG label: " + dataset.annotation_categories[label])
    plt.suptitle(text)
    plt.show()

# printNImageCompare(img, lista, n, dataset)
# IDImgs:   list of IDs of img to print
# dataset:  dataset to get the img from id
#
# prints N image in 1 column, print img number(position) and label for each image
def printNIMG(IDImgs,dataset):
    n=len(IDImgs)
    f, axarr = plt.subplots(n)
    for i in range(n):
        img, label = dataset[IDImgs[i]]
        axarr[i].imshow(img)
        axarr[i].set_title("img label: " + str(label) + " img num: " + str(i + 1))
    plt.show()

# printNImageCompare(img, lista, n, dataset)
# IDImg:    ID image to be printed in the first row
# IDImgs:   list of IDs of img for the second row
# dataset:  dataset to get the img from id
#
# prints 2 lines, the first containing img and the second containing the first n img of the list
def printNImageCompare(IDImg, IDImgs, dataset):
    f, axarr = plt.subplots(2, len(IDImgs))

    img_query, label_query = dataset[IDImg]
    axarr[0][0].imshow(img_query)
    axarr[0][0].set_title("immagine scelta")
    for i in range(len(IDImgs)):
        imgRes, labelRes = dataset[IDImgs[i]]
        axarr[1][i].imshow(imgRes)
        axarr[1][i].set_title("img num: " + str(i+1))
    plt.show()

def saveOnFileLatentFeatures(name,featuresLatenti,ID_space,redDimID):
    name = name
    if ID_space == 1:
        name+="_Layer3"
    elif ID_space == 2:
        name+="_AVGPool"
    elif ID_space == 3:
        name+="_VectorLast"

    if redDimID == 1:
        name+="_PCA"
    elif redDimID == 2:
        name+="_SVD"
    elif redDimID == 3:
        name += "_LDA"
    elif redDimID == 4:
        name += "_KMeans"
    elif redDimID == 5:
        name += "_CP"

    print("salvataggio su File")
    file = open('LatentFeatures\\' + name + '.csv', 'w', newline='')
    writer = csv.writer(file, delimiter=';')

    for row in featuresLatenti:
        writer.writerow(row)