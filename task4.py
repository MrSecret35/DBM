from scipy.special import softmax
import numpy

import genericFunction as GF
import database as DBFunc
import task2
import task3
from task1 import Task1
def processing(Caltech101,ReteNeurale):
    task1 = Task1()

    # prendere DB
    ID_space=DBFunc.IDSpace()
    DB = DBFunc.getDB(ID_space)

    # chiedere l'immagine di query
    ID_img_query = GF.getIDImg(Caltech101)
    QueryVector = task1.getVectorbyID(ID_img_query, Caltech101, ReteNeurale, ID_space)

    #chidere che metodo utilizzare
    ID_method = getIDMethod()

    #calcolare la classe
    classID = -1;
    if ID_method == 1:
        classID= getCIDAVG(QueryVector ,Caltech101, DB)
    elif ID_method == 2:
        classID = getCIDOneNN(QueryVector, Caltech101, DB)
    elif ID_method == 3:
        k = 20
        classID = getCIDKNN(QueryVector, Caltech101, DB, k)
    elif ID_method == 4:
        k = 20
        classID = getCIDKNN_importance(QueryVector, Caltech101, DB, k)

    GF.printIMG(ID_img_query,"Label ottenuta: " + Caltech101.annotation_categories[classID],Caltech101)

def getIDMethod():
    menu = {}
    menu['1'] = "AVG Feature Classified"
    menu['2'] = "1_NN"
    menu['3'] = "K_NN"
    menu['4'] = "K_NN with importance"
    options = menu.keys()
    for entry in options:
        print(entry, menu[entry])
    res= int(input("insert choice"))
    return res

def getCIDAVG(imgQueryVector, dataset, DB):
    DBLabelAVG = task3.getDatasetLabel(dataset, DB)
    for i in DBLabelAVG:
        vectorLeader = DBLabelAVG[i][0]
        for img in DBLabelAVG[i]:
            vectorLeader += img
            vectorLeader = [x / 2 for x in vectorLeader]
        DBLabelAVG[i] = vectorLeader

    simList = task2.getSimilarityVector(imgQueryVector, DBLabelAVG)
    simList = sorted(simList, key=lambda tup: tup[1])

    return int(simList[0][0]-1)

def getCIDOneNN(imgQueryVector, dataset, DB):

    simList = task2.getSimilarityVector(imgQueryVector, DB)
    simList = sorted(simList, key=lambda tup: tup[1])

    img, label= dataset[ DBFunc.getIDfromRow(simList[0][0]) ]
    return label

def getCIDKNN(imgQueryVector, dataset, DB, k):
    simList = task2.getSimilarityVector(imgQueryVector, DB)
    simList = sorted(simList, key=lambda tup: tup[1])


    img, label= dataset[ DBFunc.getIDfromRow(simList[0][0]) ]
    listLabel= []
    for i in range(k):
        img, label = dataset[DBFunc.getIDfromRow(simList[i][0])]
        listLabel.append(label)

    return findMajorityElement(listLabel)


def findMajorityElement(nums):
    m = -1
    i = 0
    for j in range(len(nums)):
        if i == 0:
            m = nums[j]
            i = 1
        elif m == nums[j]:
            i = i + 1
        else:
            i = i - 1
    return m

def getCIDKNN_importance(imgQueryVector, dataset, DB, k):
    simList = task2.getSimilarityVector(imgQueryVector, DB)
    simList = sorted(simList, key=lambda tup: tup[1])

    simList= simList[0:k]
    #take only the distance
    simListValue= [j for (i,j) in simList]
    simListValue = softmax([i.item() for i in simListValue])

    #remake the list with tuple: label, opposite of softmax distance(importance)
    for i in range(k):
        img, label = dataset[DBFunc.getIDfromRow(simList[i][0])]
        simList[i] = (label,(1- simListValue[i]) )

    #list whit unique label
    simListRes = []
    #list with the corresponding sum of importance
    simListResValue = []
    for pair in simList:
        if(pair[0] not in simListRes):
            simListRes.append(pair[0])
            sumEq= sum([j if i==pair[0] else 0 for (i,j) in simList])
            simListResValue.append(sumEq)


    return simListRes[numpy.array(simListResValue).argmax()]





