from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.decomposition import LatentDirichletAllocation as LDA

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
    DBDistance = DBFunc.getDistanceDB(1,ID_space)

    featuresLatenti = None
    if redDimID == 1:
        featuresLatenti = task6.get_PCA(DBDistance,k)
    elif redDimID == 2:
        featuresLatenti = task6.get_SVD(DBDistance, k)
    elif redDimID == 3:
        featuresLatenti = task6.get_LDA(DBDistance, k)
    elif redDimID == 4:
        featuresLatenti = task6.get_KMeans(DBDistance, k)


    featuresLatentiShape = task6.shapeLatentFeatures(featuresLatenti, k, id_row)

    GF.saveOnFileLatentFeatures("featuresKTask9", featuresLatentiShape, ID_space, redDimID)

def get_PCA(DB,k):
    pca = PCA()
    pca.fit(DB)
    resPCA= pca.transform(DB)

    eigthValue = pca.explained_variance_ratio_
    eigthValue = zip([z for z in range(len(eigthValue))], eigthValue)
    eigthValue = sorted(eigthValue, key=lambda tup: tup[1], reverse=True)
    eigthValue = eigthValue[0:k]
    eigthValue = [i for (i,j) in eigthValue]

    res=[]
    for x in resPCA:
        d= []
        for i in range(len(eigthValue)):
            d.append(x[i])
        res.append(d)

    return res

