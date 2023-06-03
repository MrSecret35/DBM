import csv
import numpy

import database as DBFunc
import task2

def processing(dataset, ReteNeurale):
    # take number of label
    N_etichetta = int(input("inserisci l'etichetta (tra 0 e 94): "))

    # take DB
    ID_space = DBFunc.IDSpace()
    DB = DBFunc.getDB(ID_space)
    id_row = DBFunc.getDBID()

    # DB Label-Features
    DictDBLabel = getDictDatasetLabel(dataset,DB, id_row)
    DBLabel = shapeDBLabel(DictDBLabel)

    # list of similarities(label) for my label
    simList = task2.getSimilarityVector(DBLabel[N_etichetta],DBLabel)
    simList = sorted(simList, key=lambda tup: tup[1])

    #Print results
    print("etichetta: ", N_etichetta, "quella più simile: ", int(simList[1][0]-1))
    print("classe: ",
          dataset.annotation_categories[N_etichetta],
          "quella più simile: ",
          dataset.annotation_categories[int(simList[1][0]-1)])

# getDictDatasetLabel(dataset,DB,id_row)
# dataset: dataset of images
# DB: matrix Image-Features DB[i][j]= value for feature j in img i
# id_row: Row-ID matching matrix (row in matrix code - ID in dataset)
#
# returns a dictionary with: for each label all the corresponding images
def getDictDatasetLabel(dataset,DB,id_row):
    res= {}
    for i in range(len(DB)):
        id= DBFunc.getIDfromRow(i+1, id_row)
        img, label = dataset[id]
        if label not in res.keys():
            res[label]= []

        res[label].append(DB[i])
    return res

# shapeDBLabel(dataset):
# dataset: dataset of images
#
# calculates the leader for each label by averaging the values(features) of the images
# return matrix Label-Features
def shapeDBLabel(dataset):
    res=[]
    for i in dataset:
        vectorLeader = dataset[i][0]
        for img in dataset[i]:
            vectorLeader += img
            vectorLeader = [x/2 for x in vectorLeader]
        res.append(vectorLeader)
    return res
