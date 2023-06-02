from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.cluster import KMeans

import database as DBFunc
import genericFunction as GF
from task1 import Task1
import task2
import task3
import task6

def processing(Caltech101,ReteNeurale):
    task1 = Task1()

    ID_img_query = GF.getIDImg(Caltech101)
    # ask N image to print
    n = int(input("insert n (numero immagini):"))

    # prendere DB
    ID_Task= DBFunc.IDTaskIMG()
    ID_space = DBFunc.IDSpace()
    ID_Dec = task6.getRedDim()

    DB = DBFunc.getDB(ID_space)
    id_row = DBFunc.getDBID()
    DBLatent = DBFunc.getLatentDB(ID_Task,ID_space,ID_Dec)

    # ask what latent features
    nk = int(input("insert la feature latente (1 ... " + str(len(DBLatent)) + "):"))
    nk= nk-1;

    QueryVector = task1.getVectorbyID(ID_img_query, Caltech101, ReteNeurale, ID_space)
    QueryVector = getQueryVectorLatent(DB,QueryVector,ID_Dec,len(DBLatent))

    QueryWeight= QueryVector[nk]
    DBLatent = DBLatent[nk]

    listaSim= getSimilarityVector(QueryWeight,DBLatent)

    #listaSim = [i for (i,j) in listaSim]
    img_query, label_query = Caltech101[ID_img_query]

    printNImage(img_query, listaSim, n, Caltech101, id_row)


def getQueryVectorLatent(DB,QueryVector,ID_Dec,k):
    if ID_Dec == 1:
        featuresLatenti = get_PCA(DB,QueryVector,k)
    elif ID_Dec == 2:
        featuresLatenti = get_SVD(DB,QueryVector,k)
    elif ID_Dec == 3:
        featuresLatenti = get_LDA(DB,QueryVector,k)
    elif ID_Dec == 4:
        featuresLatenti = get_KMeans(DB,QueryVector,k)

    return featuresLatenti[0]

def get_SVD(DB,QueryVector,k):
    u, s, vh = np.linalg.svd([QueryVector], full_matrices=True)

    u = [ u[i][0:k] for i in range(len(u))]
    return u
def get_PCA(DB,QueryVector,k):
    pca = PCA(n_components=k)
    pca.fit(DB)
    res= pca.transform([QueryVector])

    return res

def get_LDA(DB,QueryVector,k):
    lda = LDA(n_components=k)
    argMin= abs(getArgMin(DB))
    DB_p = [[DB[i][j]+argMin for j in range(len(DB[i]))] for i in range(len(DB))]
    lda.fit(DB_p)
    res= lda.transform([QueryVector])

    return res

def getArgMin(DB):
    argmin= DB[0][0]
    for i in range(len(DB)):
        if min(DB[i])<argmin:
            argmin=min(DB[i])
    return argmin

def get_KMeans(DB,QueryVector,k):
    kmeans = KMeans(n_clusters=k, n_init='auto')
    kmeans.fit(DB)
    res = kmeans.transform([QueryVector])

    return res

def getSimilarityVector(QueryWeight,DBLatent):
    for img in DBLatent:
        DBLatent[1] = abs(QueryWeight - DBLatent[1])
    return sorted(DBLatent, key=lambda tup: tup[1])

def printNImage(img, list, n, dataset,id_row):
    f, axarr = plt.subplots(2, n)
    axarr[0][0].imshow(img)
    axarr[0][0].set_title("immagine scelta")
    for i in range(n):
        imgRes, labelRes = dataset[ list[i][0]]
        axarr[1][i].imshow(imgRes)
        axarr[1][i].set_title("img num: " + str(i+1))
    plt.show()