from scipy.special import softmax
import numpy

import genericFunction as GF
import database as DBFunc
from task1 import Task1
import task2
import task3

def processing(Caltech101,ReteNeurale):
    task1 = Task1()

    # take DB:
    ID_space=DBFunc.IDSpace()
    DB = DBFunc.getDB(ID_space)
    id_row = DBFunc.getDBID()

    # take IMG for the query
    ID_img_query = GF.getIDImg(Caltech101)
    QueryVector = task1.getVectorbyID(ID_img_query, Caltech101, ReteNeurale, ID_space)

    # take ID of method
    ID_method = getIDMethod()

    # calculates the class corresponding to the query image
    classID = -1
    if ID_method == 1:
        classID= getCIDAVG(QueryVector ,Caltech101, DB, id_row)
    elif ID_method == 2:
        classID = getCIDOneNN(QueryVector, Caltech101, DB, id_row)
    elif ID_method == 3:
        k = 20
        classID = getCIDKNN(QueryVector, Caltech101, DB, id_row, k)
    elif ID_method == 4:
        k = 20
        classID = getCIDKNN_importance(QueryVector, Caltech101, DB, id_row, k)

    # print results
    GF.printIMG(ID_img_query,"Label ottenuta: " + Caltech101.annotation_categories[classID],Caltech101)

# getIDMethod()
#
# ask and return an ID for method
def getIDMethod():
    menu = {}
    menu['1'] = "AVG Feature Classified"
    menu['2'] = "1_NN"
    menu['3'] = "K_NN"
    menu['4'] = "K_NN with importance"
    options = menu.keys()
    for entry in options:
        print(entry, menu[entry])
    res= int(input("insert choice: "))
    return res

# getCIDAVG(imgQueryVector, dataset, DB, id_row)
# imgQueryVector: feature vector for the query image
# dataset: dataset of images
# DB: matrix Image-Features DB[i][j]= value for feature j in img i
# id_row: Row-ID matching matrix (row in matrix code - ID in dataset)
#
# calculates the class corresponding to the query image with similarity in Label-Feature DB
# return ID of Class/Label
def getCIDAVG(imgQueryVector, dataset, DB, id_row):
    DictDBLabelAVG = task3.getDictDatasetLabel(dataset, DB, id_row)
    DBLabelAVG = task3.shapeDBLabel(DictDBLabelAVG)

    simList = task2.getSimilarityVector(imgQueryVector, DBLabelAVG)
    simList = sorted(simList, key=lambda tup: tup[1])

    return int(simList[0][0]-1)

# getCIDAVG(imgQueryVector, dataset, DB, id_row)
# imgQueryVector: feature vector for the query image
# dataset: dataset of images
# DB: matrix Image-Features DB[i][j]= value for feature j in img i
# id_row: Row-ID matching matrix (row in matrix code - ID in dataset)
#
# calculates the class corresponding to the query image associating the class of the closest img in the DB
# return ID of Class/Label
def getCIDOneNN(imgQueryVector, dataset, DB,id_row):

    simList = task2.getSimilarityVector(imgQueryVector, DB)
    simList = sorted(simList, key=lambda tup: tup[1])

    img, label= dataset[ DBFunc.getIDfromRow(simList[0][0],id_row) ]
    return label

# getCIDAVG(imgQueryVector, dataset, DB, id_row)
# imgQueryVector: feature vector for the query image
# dataset: dataset of images
# DB: matrix Image-Features DB[i][j]= value for feature j in img i
# id_row: Row-ID matching matrix (row in matrix code - ID in dataset)
# k: range of images
#
# calculates the class corresponding to the query image associating the most common class (majority) in a range k
# return ID of Class/Label
def getCIDKNN(imgQueryVector, dataset, DB, id_row, k):
    simList = task2.getSimilarityVector(imgQueryVector, DB)
    simList = sorted(simList, key=lambda tup: tup[1])


    img, label= dataset[ DBFunc.getIDfromRow(simList[0][0],id_row) ]
    listLabel= []
    for i in range(k):
        img, label = dataset[DBFunc.getIDfromRow(simList[i][0],id_row)]
        listLabel.append(label)

    return findMajorityElement(listLabel)


# findMajorityElement(nums)
# nums: vector of number
#
# return/find the element with the most appearances in the vector
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

# getCIDAVG(imgQueryVector, dataset, DB, id_row)
# imgQueryVector: feature vector for the query image
# dataset: dataset of images
# DB: matrix Image-Features DB[i][j]= value for feature j in img i
# id_row: Row-ID matching matrix (row in matrix code - ID in dataset)
# k: range of images
#
# calculates the class corresponding to the query image associating the most voted class (importance/proximity) in a range k
# return ID of Class/Label
def getCIDKNN_importance(imgQueryVector, dataset, DB, id_row, k):
    simList = task2.getSimilarityVector(imgQueryVector, DB)
    simList = sorted(simList, key=lambda tup: tup[1])

    simList= simList[0:k]
    #take only the distance
    simListValue= [j for (i,j) in simList]
    simListValue = softmax([i.item() for i in simListValue])

    #remake the list with tuple: label, opposite of softmax distance(importance)
    for i in range(k):
        img, label = dataset[DBFunc.getIDfromRow(simList[i][0], id_row)]
        simList[i] = (label,(1 - simListValue[i]) )

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





