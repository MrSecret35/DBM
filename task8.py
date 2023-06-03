from tqdm import tqdm

import database as DBFunc
import genericFunction as GF
import task2
import task3
import task6
def processing(Caltech101,ReteNeurale):
    # prendere DB
    ID_space = DBFunc.IDSpace()
    id_row = DBFunc.getDBID()

    # prendere k (features latenti)
    k = int(input("inserisci k (features latenti): "))

    # prendere tecnica
    redDimID = task6.getRedDim()

    # take DB
    DB = DBFunc.getDB(ID_space)

    DBLabelDistance = createDBLabelLabel(Caltech101, DB, id_row)

    # featuresLatenti = lista di obj, per ogni obj ha una lista con i k valori per le k feature latenti
    featuresLatenti = None
    if redDimID == 1:
        featuresLatenti = task6.get_PCA(DBLabelDistance, k)
    elif redDimID == 2:
        featuresLatenti = task6.get_SVD(DBLabelDistance, k)
    elif redDimID == 3:
        featuresLatenti = task6.get_LDA(DBLabelDistance, k)
    elif redDimID == 4:
        featuresLatenti = task6.get_KMeans(DBLabelDistance, k)

    featuresLatentiShape = shapeLatentFeatures(featuresLatenti, k, id_row)

    GF.saveOnFileLatentFeatures("featuresKTask8",featuresLatentiShape,ID_space,redDimID)

def createDBLabelLabel(Caltech101, DB, id_row):
    DictDBLabel = task3.getDictDatasetLabel(Caltech101, DB, id_row)
    DBLabel = task3.shapeDBLabel(DictDBLabel)
    DBLabelDistance = calculateLabelDistanceDB(DBLabel)

    return DBLabelDistance

def calculateLabelDistanceDB(dataset):
    res= []
    for label in tqdm(dataset):
        distanze= task2.getSimilarityVector(label,dataset)
        distanze= [j.detach().numpy().item() for (i,j) in distanze]
        res.append(distanze)
    return res

def shapeLatentFeatures(featuresLatenti,k,id_row):
    featuresLatentiShape = []
    for i in tqdm(range(k)):
        d = []
        for j in range(len(featuresLatenti)):
            d.append((j, featuresLatenti[j][i]))
        d = sorted(d, key=lambda tup: tup[1], reverse=True)
        featuresLatentiShape.append(d)

    return featuresLatentiShape