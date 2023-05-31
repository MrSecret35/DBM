import numpy as np
from sklearn.decomposition import PCA
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.cluster import KMeans
import csv
from tqdm import tqdm

import database as DBFunc
import genericFunction as GF

def processing(Caltech101,ReteNeurale):
    # prendere DB
    ID_space = DBFunc.IDSpace()
    id_row= DBFunc.getDBID()

    # prendere k (features latenti)
    k = int(input("inserisci k (features latenti): "))

    # prendere tecnica
    redDimID = getRedDim()

    # take DB
    DB = DBFunc.getDB(ID_space)

    #featuresLatenti = lista di obj, per ogni obj ha una lista con i k valori per le k feature latenti
    featuresLatenti = None
    if redDimID == 1:
        featuresLatenti = get_PCA(DB,k)
    elif redDimID == 2:
        featuresLatenti = get_SVD(DB,k)
    elif redDimID == 3:
        featuresLatenti = get_LDA(DB,k)
    elif redDimID == 4:
        featuresLatenti = get_KMeans(DB,k)

    #re shape featuresLatenti come lista di feature, ogni feature è una lista di coppie (ID obj, valore)
    featuresLatentiShape = []
    for i in tqdm(range(k)):
        d= []
        for j in range(len(featuresLatenti)):
            d.append( (DBFunc.getIDfromRow(j+1,id_row) , featuresLatenti[j][i]) )
        featuresLatentiShape.append(d)

    saveFeaturesLatenti(featuresLatentiShape,ID_space,redDimID)


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

def getArgMin(DB):
    argmin= DB[0][0]
    for i in range(len(DB)):
        if min(DB[i])<argmin:
            argmin=min(DB[i])
    return argmin

def get_KMeans(DB,k):
    kmeans = KMeans(n_clusters=k, n_init='auto')
    kmeans.fit(DB)
    res = kmeans.transform(DB)

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
