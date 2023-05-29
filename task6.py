import numpy as np
from sklearn.decomposition import PCA
from sklearn.decomposition import LatentDirichletAllocation as LDA

import database as DBFunc
import genericFunction as GF

def processing(Caltech101,ReteNeurale):
    # prendere DB
    ID_space = DBFunc.IDSpace()

    # prendere k (features latenti)
    k = int(input("inserisci k (features latenti): "))

    #prendere tecnica
    redDimID = getRedDim()

    # take DB
    DB = DBFunc.getDB(ID_space)

    featuresLatenti = None
    if redDimID == 1:
        featuresLatenti = get_PCA(DB,k)
    elif redDimID == 2:
        featuresLatenti = get_SVD(DB,k)
    elif redDimID == 3:
        featuresLatenti = get_LDA(DB,k)
    elif redDimID == 4:
        print("ciao")

    print(featuresLatenti)
    saveFeaturesLatenti(featuresLatenti)


def getRedDim():
    redDim = ''
    while redDim != '1' and redDim != '2' and redDim != '3' and redDim != '4':
        redDim = input("select space: \n 1 - PCA\n 2 - SVD\n 3 - LDA\n 4 - k-means\n")
        if (redDim != '1' and redDim != '2' and redDim != '3' and redDim != '4'):
            print("insert a valid selection")
    return int(redDim)

def get_SVD(DB,k):
    u, s, vh = np.linalg.svd(DB, full_matrices=True)

    u = [ u[i][0:k] for i in range(len(u))]
    return u

def get_PCA(DB,k):
    pca = PCA(n_components=k)
    pca.fit(DB)
    res= pca.transform(DB)

    return res

def get_LDA(DB,k):
    lda = LDA(n_components=k)
    argMin= abs(getArgMin(DB))
    DB_p = [[DB[i][j]+argMin for j in range(len(DB[i]))] for i in range(len(DB))]
    lda.fit(DB_p)
    res= lda.transform(DB_p)

    return res

def saveFeaturesLatenti(featuresLatenti):
    print("salva")

def getArgMin(DB):
    argmin= DB[0][0]
    for i in range(len(DB)):
        if min(DB[i])<argmin:
            argmin=min(DB[i])
    return argmin