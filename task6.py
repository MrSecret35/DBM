import numpy as np
from sklearn.decomposition import PCA
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.cluster import KMeans
import csv
from tqdm import tqdm

import database as DBFunc
import genericFunction as GF

def processing(dataset,ReteNeurale):

    # take DB
    ID_space = DBFunc.IDSpace()
    DB = DBFunc.getDB(ID_space)
    id_row= DBFunc.getDBID()

    # take k (latent features)
    k = int(input("inserisci k (features latenti): "))

    # take id of method
    redDimID = getRedDim()

    # featuresLatenti = list of obj, for each obj it has a list with the k values for the k latent features
    featuresLatenti = None
    if redDimID == 1:
        featuresLatenti = get_PCA(DB,k)
    elif redDimID == 2:
        featuresLatenti = get_SVD(DB,k)
    elif redDimID == 3:
        featuresLatenti = get_LDA(DB,k)
    elif redDimID == 4:
        featuresLatenti = get_KMeans(DB,k)

    # reshape featuresLatenti come as a list of features, each feature is a list of pairs (id obj, value)
    featuresLatentiShape= shapeLatentFeatures(featuresLatenti,k,id_row)

    # save latent features on file
    GF.saveOnFileLatentFeatures("featuresKTask6",featuresLatentiShape,ID_space,redDimID)

# getRedDim()
#
# ask and return an ID for method
def getRedDim():
    redDim = ''
    while redDim != '1' and redDim != '2' and redDim != '3' and redDim != '4':
        redDim = input("select space: \n 1 - PCA\n 2 - SVD\n 3 - LDA\n 4 - k-means\n")
        if (redDim != '1' and redDim != '2' and redDim != '3' and redDim != '4'):
            print("insert a valid selection")
    return int(redDim)

# get_SVD(DB,k)
# DB: Database/matrix Image-Features
# k: number of latent features
#
# take k latent feature with SVD decomposition
def get_SVD(DB,k):
    u, s, vh = np.linalg.svd(DB, full_matrices=True)

    u = [ u[i][0:k] for i in range(len(u))]
    return u

# get_PCA(DB,k)
# DB: Database/matrix Image-Features
# k: number of latent features
#
# take k latent feature with PCA decomposition
def get_PCA(DB,k):
    pca = PCA(n_components=k)
    pca.fit(DB)
    res= pca.transform(DB)

    return res

# get_LDA(DB,k)
# DB: Database/matrix Image-Features
# k: number of latent features
#
# take k latent feature with LatentDirichletAllocation
def get_LDA(DB,k):
    lda = LDA(n_components=k)
    argMin= abs(getArgMin(DB))
    # brings all values ​​to positive
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

# get_KMeans(DB,k)
# DB: Database/matrix Image-Features
# k: number of latent features
#
# take k latent feature with KMeans, k distance from k cluster centroid
def get_KMeans(DB,k):
    kmeans = KMeans(n_clusters=k, n_init='auto')
    kmeans.fit(DB)
    res = kmeans.transform(DB)

    return res

# shapeLatentFeatures(featuresLatenti,k,id_row)
# featuresLatenti: list of obj, for each obj it has a list with the k values for the k latent features
# k: number of latent features
# id_row: Row-ID matching matrix (row in matrix code - ID in dataset)
#
# reshape featuresLatenti come as a list of features, each feature is a list of pairs (id obj, value), each feature is sorted in descending order
def shapeLatentFeatures(featuresLatenti,k,id_row):
    featuresLatentiShape = []
    for i in tqdm(range(k)):
        d = []
        for j in range(len(featuresLatenti)):
            d.append((DBFunc.getIDfromRow(j + 1, id_row), featuresLatenti[j][i]))
        d = sorted(d, key=lambda tup: tup[1], reverse=True)
        featuresLatentiShape.append(d)

    return featuresLatentiShape

