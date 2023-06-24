import tensorly as tl
from tensorly.decomposition import parafac
import numpy

import database as DBFunc
import genericFunction as GF
from task1 import Task1
import task2
import task3
import task4
import task6
import task10
def processing(dataset,ReteNeurale):
    task1 = Task1()

    # take ID of IMG for the query
    ID_img_query = GF.getIDImg(dataset)

    # take DB
    ID_Task= DBFunc.IDTask()
    ID_space = DBFunc.IDSpace()
    ID_Dec=5
    if(ID_Task != 7):
        ID_Dec = task6.getRedDim()

    DB = DBFunc.getDB(ID_space)
    id_row = DBFunc.getDBID()

    DBLatent = DBFunc.getLatentDB(ID_Task, ID_space, ID_Dec)


    QueryVector = task1.getVectorbyID(ID_img_query, dataset, ReteNeurale, ID_space)
    if(ID_Task==7):
        QueryVector=getQueryVectorLatent(DB,QueryVector,ID_Dec,len(DBLatent))
    else:
        QueryVector = task10.getQueryVectorLatent(DB, QueryVector, ID_Dec, len(DBLatent))

    DBLatent = task10.shapeDBLatent(DBLatent)
    classID=-1
    if ID_Task==7 or ID_Task==8 :
        classID = getCIDAVG(QueryVector,DBLatent)
    elif ID_Task==6 or ID_Task==9:
        k=20
        classID = task4.getCIDKNN_importance(QueryVector, dataset, DBLatent, id_row, k)

    print(classID)
    # Print results
    GF.printIMG(ID_img_query,"Label ottenuta: " + dataset.annotation_categories[classID],dataset)

def getQueryVectorLatent(DB,QueryVector,ID_Dec,k):
    if ID_Dec == 5:

        DB_tensor = tl.tensor(DB)
        # take k latent feature with Parafac decomposition
        (weights, factors) = parafac(DB_tensor, rank=k)

        featuresLatentiFeatures= factors[1]
        featuresLatentiFeatures = numpy.matrix(featuresLatentiFeatures).T
        QueryVector = numpy.matrix(QueryVector).T

        QueryVector = numpy.array(featuresLatentiFeatures.dot(QueryVector).T)[0]
    return QueryVector
def getCIDAVG(imgQueryVector, DB):
    simList = task2.getSimilarityVector(imgQueryVector, DB)
    simList = sorted(simList, key=lambda tup: tup[1])

    return int(simList[0][0]-1)