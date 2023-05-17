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
        print("ciao")
        classID= getCIDAVG(QueryVector ,Caltech101, DB)
    elif ID_method == 2:
        print("ciao")
    elif ID_method == 3:
        print("ciao")

    GF.printIMG(ID_img_query,"Label ottenuta: " + Caltech101.annotation_categories[classID],Caltech101)

def getIDMethod():
    menu = {}
    menu['1'] = "AVG Feature Classified"
    menu['2'] = "1_NN"
    menu['3'] = "K_NN"
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

    return int(simList[0][0]/2)

