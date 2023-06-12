import csv
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.cluster import KMeans

import database as DBFunc
import genericFunction as GF
from task1 import Task1
import task2
import task3
import task6
import task10
def processing(dataset,ReteNeurale):

    # take DB
    id_row = DBFunc.getDBID()


    for ID_space in [1,2,3]:
        for ID_Dec in [1,2,3,4]:
            for ID_Task in [6,9]:
                # take latentDB
                DBLatent = DBFunc.getLatentDB(ID_Task, ID_space, ID_Dec)
                DBLatent = task10.shapeDBLatent(DBLatent)
                # calculate distance  # save distance
                file = takeFile(ID_Task, ID_space, ID_Dec)
                writerDistance = csv.writer(file, delimiter=';')
                for img in tqdm(DBLatent):
                    distanze = task2.getSimilarityVector(img, DBLatent)
                    distanze = [j.detach().numpy().item() for (i, j) in distanze]
                    writerDistance.writerow(distanze)



def takeFile(ID_Task,ID_space,ID_Dec):
    fileDir = "Distance/EuclideanDistance_"
    if ID_Task == 6:
        fileDir += "featuresKTask6"
    elif ID_Task == 9:
        fileDir += "featuresKTask9"

    if ID_space == 1:
        fileDir += "_Layer3"
    elif ID_space == 2:
        fileDir += "_AVGPool"
    elif ID_space == 3:
        fileDir += "_VectorLast"

    if ID_Dec == 1:
        fileDir += "_PCA"
    elif ID_Dec == 2:
        fileDir += "_SVD"
    elif ID_Dec == 3:
        fileDir += "_LDA"
    elif ID_Dec == 4:
        fileDir += "_KMeans"

    fileDir += ".csv"
    file = open(fileDir, 'w',newline='')

    return file