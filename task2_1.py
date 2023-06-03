import csv
from tqdm import tqdm

import database as DBFunc
import task2
def processing(dataset, ReteNeurale):

    # ask ID spase
    ID_Space = DBFunc.IDSpace()

    # take DB: matrix corresponding to the csv file indicated by the ID_space
    DB = DBFunc.getDB(ID_Space)

    #ID od what distance use
    ID_Distance = takeIDDistance()

    fileDistance = takeFileDistance(ID_Distance,ID_Space)
    writerDistance = csv.writer(fileDistance, delimiter=';')

    for img in tqdm(DB):
        distanze = task2.getSimilarityVector(img,DB)
        distanze = [j.detach().numpy().item() for (i,j) in distanze]
        writerDistance.writerow(distanze)


def takeIDDistance():
    ID_Distance = ''
    while ID_Distance != '1':
        ID_Distance = input("select space: \n 1 - euclidean \nChoice:  ")
        if (ID_Distance != '1'):
            print("insert a valid selection")
    return int(ID_Distance)


def takeFileDistance(ID_Distance,ID_Space):
    fileDistance = {}
    if ID_Distance == 1:
        if ID_Space == 1:
            fileDistance = open('Distance\EuclideanDistance_Layer3.csv', 'w', newline='')
        elif ID_Space == 2:
            fileDistance = open('Distance\EuclideanDistance_AVGPool.csv', 'w',newline='')
        elif ID_Space == 3:
            fileDistance = open('Distance\EuclideanDistance_Last.csv', 'w', newline='')

    return fileDistance
