import tensorly as tl
from tensorly.decomposition import parafac

import database as DBFunc
import genericFunction as GF
import task3
import task6
import task7
import task8
import task9
def processing(dataset,ReteNeurale):
    print("ciao")
    IDTask = ID_Task()

    # take k (latent features)
    k = int(input("inserisci k (features latenti): "))

    if IDTask == 6:
        task_6(dataset,k)
    elif IDTask == 7:
        task_7(dataset,k)
    elif IDTask == 8:
        task_8(dataset, k)
    elif IDTask == 9:
        task_9(dataset, k)


def ID_Task():
    ID_task = ''
    while ID_task != '6' and ID_task != '7' and ID_task != '8' and ID_task != '9':
        ID_task = input("select space: \n 6 - task6\n 7 - task7\n 8 - task8\n 9 - task9\n")
        if (ID_task != '6' and ID_task != '7' and ID_task != '8' and ID_task != '9'):
            print("insert a valid selection")
    return int(ID_task)


def task_6(dataset, k):
    id_row = DBFunc.getDBID()
    for i in range(3):
        DB = DBFunc.getDB(i + 1)
        featuresLatenti = task6.get_PCA(DB, k)
        featuresLatentiShape = task6.shapeLatentFeatures(featuresLatenti, k, id_row)
        GF.saveOnFileLatentFeatures("featuresKTask6", featuresLatentiShape, i + 1, 1)

    for i in range(3):
        DB = DBFunc.getDB(i + 1)
        featuresLatenti = task6.get_SVD(DB, k)
        featuresLatentiShape = task6.shapeLatentFeatures(featuresLatenti, k, id_row)
        GF.saveOnFileLatentFeatures("featuresKTask6", featuresLatentiShape, i + 1, 2)

    for i in range(3):
        DB = DBFunc.getDB(i + 1)
        featuresLatenti = task6.get_LDA(DB, k)
        featuresLatentiShape = task6.shapeLatentFeatures(featuresLatenti, k, id_row)
        GF.saveOnFileLatentFeatures("featuresKTask6", featuresLatentiShape, i + 1, 3)

    for i in range(3):
        DB = DBFunc.getDB(i + 1)
        featuresLatenti = task6.get_KMeans(DB, k)
        featuresLatentiShape = task6.shapeLatentFeatures(featuresLatenti, k, id_row)
        GF.saveOnFileLatentFeatures("featuresKTask6", featuresLatentiShape, i + 1, 4)

def task_7(dataset, k):
    id_row = DBFunc.getDBID()
    for i in range(3):
        DB = DBFunc.getDB(i + 1)
        DBLabel = task3.getDictDatasetLabel(dataset, DB, id_row)
        DB = task3.shapeDBLabel(DBLabel)
        DB_tensor = tl.tensor(DB)
        (weights, factors) = parafac(DB_tensor, rank=k)
        featuresLatentiShape = task7.shapeLatentFeatures(factors[0], k, id_row)
        GF.saveOnFileLatentFeatures("featuresKTask7", featuresLatentiShape, i + 1, 5)

def task_8(dataset, k):
    id_row = DBFunc.getDBID()
    for i in range(3):
        DB = DBFunc.getDB(i + 1)
        DB = task8.createDBLabelLabel(dataset, DB, id_row)
        featuresLatenti = task6.get_PCA(DB, k)
        featuresLatentiShape = task7.shapeLatentFeatures(featuresLatenti, k, id_row)
        GF.saveOnFileLatentFeatures("featuresKTask8", featuresLatentiShape, i + 1, 1)

    for i in range(3):
        DB = DBFunc.getDB(i + 1)
        DB = task8.createDBLabelLabel(dataset, DB, id_row)
        featuresLatenti = task6.get_SVD(DB, k)
        featuresLatentiShape = task7.shapeLatentFeatures(featuresLatenti, k, id_row)
        GF.saveOnFileLatentFeatures("featuresKTask8", featuresLatentiShape, i + 1, 2)

    for i in range(3):
        DB = DBFunc.getDB(i + 1)
        DB = task8.createDBLabelLabel(dataset, DB, id_row)
        featuresLatenti = task6.get_LDA(DB, k)
        featuresLatentiShape = task7.shapeLatentFeatures(featuresLatenti, k, id_row)
        GF.saveOnFileLatentFeatures("featuresKTask8", featuresLatentiShape, i + 1, 3)

    for i in range(3):
        DB = DBFunc.getDB(i + 1)
        DB = task8.createDBLabelLabel(dataset, DB, id_row)
        featuresLatenti = task6.get_KMeans(DB, k)
        featuresLatentiShape = task7.shapeLatentFeatures(featuresLatenti, k, id_row)
        GF.saveOnFileLatentFeatures("featuresKTask8", featuresLatentiShape, i + 1, 4)

def task_9(dataset, k):
    id_row = DBFunc.getDBID()
    for i in range(3):
        DB = DBFunc.getDB(i + 1)
        DB = DBFunc.getDistanceDB(1, i+1)
        featuresLatenti = task6.get_PCA(DB, k)
        featuresLatentiShape = task6.shapeLatentFeatures(featuresLatenti, k, id_row)
        GF.saveOnFileLatentFeatures("featuresKTask9", featuresLatentiShape, i + 1, 1)

    for i in range(3):
        DB = DBFunc.getDB(i + 1)
        DB = DBFunc.getDistanceDB(1, i + 1)
        featuresLatenti = task6.get_SVD(DB, k)
        featuresLatentiShape = task6.shapeLatentFeatures(featuresLatenti, k, id_row)
        GF.saveOnFileLatentFeatures("featuresKTask9", featuresLatentiShape, i + 1, 2)

    for i in range(3):
        DB = DBFunc.getDB(i + 1)
        DB = DBFunc.getDistanceDB(1, i + 1)
        featuresLatenti = task6.get_LDA(DB, k)
        featuresLatentiShape = task6.shapeLatentFeatures(featuresLatenti, k, id_row)
        GF.saveOnFileLatentFeatures("featuresKTask9", featuresLatentiShape, i + 1, 3)

    for i in range(3):
        DB = DBFunc.getDB(i + 1)
        DB = DBFunc.getDistanceDB(1, i + 1)
        featuresLatenti = task6.get_KMeans(DB, k)
        featuresLatentiShape = task6.shapeLatentFeatures(featuresLatenti, k, id_row)
        GF.saveOnFileLatentFeatures("featuresKTask9", featuresLatentiShape, i + 1, 4)