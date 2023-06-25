import numpy
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

def processing(dataset,ReteNeurale):
    task1 = Task1()

    ID_img_query = GF.getIDImg(dataset)
    # ask N image to print
    n = int(input("insert n (numero immagini da stampare):"))

    # take DB
    ID_Task= DBFunc.IDTaskIMG()
    ID_space = DBFunc.IDSpace()
    ID_Dec = task6.getRedDim()

    DB = DBFunc.getDB(ID_space)
    id_row = DBFunc.getDBID()

    DBLatent = DBFunc.getLatentDB(ID_Task,ID_space,ID_Dec)

    QueryVector = task1.getVectorbyID(ID_img_query, dataset, ReteNeurale, ID_space)
    QueryVector = getQueryVectorLatent(DB,QueryVector,ID_Dec,len(DBLatent))

    DBLatent = shapeDBLatent(DBLatent)
    listaSim= task2.getSimilarityVector(QueryVector,DBLatent)
    listaSim= sorted(listaSim, key=lambda tup: tup[1])
    listaSim = [i for (i,j) in listaSim[0:n]]
    listID = [DBFunc.getIDfromRow(i, id_row) for i in listaSim]

    GF.printNImageCompare(ID_img_query, listID, dataset)

# getQueryVectorLatent(DB,QueryVector,ID_Dec,k)
# DB: Database/matrix Image-Latent Features
# QueryVector: vector related to query obj
# ID_Dec: id of the decomposition method
# k: number of latent features
#
# transforms the queryVector into queryVector in latent space
def getQueryVectorLatent(DB,QueryVector,ID_Dec,k):
    if ID_Dec == 1: #PCA
        pca = PCA(n_components=k)
        pca.fit(DB)

        covarianza=pca.get_covariance()
        featuresLatentiFeatures = pca.transform(covarianza)

        featuresLatentiFeatures= numpy.matrix(featuresLatentiFeatures).T
        QueryVector = numpy.matrix(QueryVector).T

        QueryVector = numpy.array(featuresLatentiFeatures.dot(QueryVector).T)[0]
    elif ID_Dec == 2: #SVD
        u, s, vh = np.linalg.svd(DB, full_matrices=True)
        featuresLatentiFeatures = vh[0:k]

        QueryVector = numpy.matrix(QueryVector).T
        QueryVector = numpy.array(featuresLatentiFeatures.dot(QueryVector).T)[0]
    elif ID_Dec == 3: #LDA
        lda = LDA(n_components=k)
        argMin = abs(task6.getArgMin(DB))
        DB_p = [[DB[i][j] + argMin for j in range(len(DB[i]))] for i in range(len(DB))]
        lda.fit(DB_p)
        featuresLatentiFeatures= lda.exp_dirichlet_component_

        QueryVector = numpy.matrix(QueryVector).T
        QueryVector = numpy.array(featuresLatentiFeatures.dot(QueryVector).T)[0]
    elif ID_Dec == 4: #KMeans
        kmeans = KMeans(n_clusters=k, n_init='auto')
        kmeans.fit(DB)

        centers= kmeans.cluster_centers_
        res= []
        for i in range(k):
            distance= task2.distance(QueryVector,centers[i])
            res.append(distance.detach().numpy().item())
        QueryVector= res

    return QueryVector

def shapeDBLatent(DBLatent):
    for i in range(len(DBLatent)):
        DBLatent[i]= sorted(DBLatent[i], key=lambda tup: tup[0])
        DBLatent[i] = [j for (i,j) in DBLatent[i]]

    DBLatent = numpy.array(DBLatent).T
    return DBLatent
