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

# prepare the function to preprocess images to be compatible with ResNet50
default_weights = ResNet50_Weights.DEFAULT
preprocess = default_weights.transforms()

def get_features(name,features):
    def hook(model, input, output):
        features[name] = output.detach()
    return hook

def IMGtoTensor(img):
  if torchvision.transforms.functional.get_image_num_channels(img) != 1:
    proc_img = preprocess(img).unsqueeze(0)
    return proc_img
  else:
    print ("incompatiable image format -- try another one")

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

def printNIMG(IDImg,dataset):
    n=len(IDImg)
    f, axarr = plt.subplots(n)
    for i in range(n):
        img, label = dataset[IDImg[i]]
        axarr[i].imshow(img)
        axarr[i].set_title("img label: " + str(label) + " img num: " + str(i + 1))
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