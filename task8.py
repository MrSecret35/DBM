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

    DB = createDBLabelLabel(Caltech101, DB, id_row)

    # featuresLatenti = lista di obj, per ogni obj ha una lista con i k valori per le k feature latenti
    featuresLatenti = None
    if redDimID == 1:
        featuresLatenti = task6.get_PCA(DB, k)
    elif redDimID == 2:
        featuresLatenti = task6.get_SVD(DB, k)
    elif redDimID == 3:
        featuresLatenti = task6.get_LDA(DB, k)
    elif redDimID == 4:
        featuresLatenti = task6.get_KMeans(DB, k)

    featuresLatentiShape = task6.shapeLatentFeatures(featuresLatenti, k, id_row)

    GF.saveOnFileLatentFeatures("featuresKTask8",featuresLatentiShape,ID_space,redDimID)

def createDBLabelLabel(Caltech101, DB, id_row):
    DBLabel = task3.getDatasetLabel(Caltech101, DB, id_row)
    DBLabel= shapeDBLabel(DBLabel)
    DBLabel = calculateDistanceDB(DBLabel)

    return DBLabel

def shapeDBLabel(dataset):
    res=[]
    for i in dataset:
        vectorLeader = dataset[i][0]
        for img in dataset[i]:
            vectorLeader += img
            vectorLeader = [x/2 for x in vectorLeader]
        res.append(vectorLeader)
    return res

def calculateDistanceDB(dataset):
    res= []
    for label in tqdm(dataset):
        distanze= task2.getSimilarityVector(label,dataset)
        distanze= [j.detach().numpy().item() for (i,j) in distanze]
        res.append(distanze)
    return res

def saveFeaturesLatenti(featuresLatenti,ID_space,redDimID):
    name = "featuresK"
    if ID_space == 1:
        name+="_Layer3_"
    elif ID_space == 2:
        name+="_AVGPool_"
    elif ID_space == 3:
        name+="_VectorLast_"

    if redDimID == 1:
        name+="_PCA"
    elif redDimID == 2:
        name+="_SVD"
    elif redDimID == 3:
        name += "_LDA"
    elif redDimID == 4:
        name += "_KMeans"

    print("salvataggio su File")
    file = open('LatentFeatures\\' + name + '.csv', 'w', newline='')
    writer = csv.writer(file, delimiter=';')

    for row in featuresLatenti:
        writer.writerow(row)