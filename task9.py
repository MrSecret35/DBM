from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.decomposition import LatentDirichletAllocation as LDA

import database as DBFunc
import genericFunction as GF
import task2
import task3
import task6

def processing(Caltech101,ReteNeurale):
    # take DB
    ID_space = DBFunc.IDSpace()
    DB = DBFunc.getDB(ID_space)
    id_row = DBFunc.getDBID()
    DBDistance = DBFunc.getDistanceDB(1, ID_space)

    # take k (latent features)
    k = int(input("inserisci k (features latenti): "))

    # take id of method
    redDimID = task6.getRedDim()

    # featuresLatenti = list of obj, for each obj it has a list with the k values for the k latent features
    featuresLatenti = None
    if redDimID == 1:
        featuresLatenti = task6.get_PCA(DBDistance,k)
    elif redDimID == 2:
        featuresLatenti = task6.get_SVD(DBDistance, k)
    elif redDimID == 3:
        featuresLatenti = task6.get_LDA(DBDistance, k)
    elif redDimID == 4:
        featuresLatenti = task6.get_KMeans(DBDistance, k)

    # reshape featuresLatenti come as a list of features, each feature is a list of pairs (id obj, value)
    featuresLatentiShape = task6.shapeLatentFeatures(featuresLatenti, k, id_row)

    # save latent features on file
    GF.saveOnFileLatentFeatures("featuresKTask9", featuresLatentiShape, ID_space, redDimID)

