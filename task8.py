from tqdm import tqdm

import database as DBFunc
import genericFunction as GF
import task2
import task3
import task6
import task7
def processing(dataset,ReteNeurale):

    # take DB
    ID_space = DBFunc.IDSpace()
    id_row = DBFunc.getDBID()
    DB = DBFunc.getDB(ID_space)

    # take id of method
    redDimID = task6.getRedDim()

    # take DB Label-Label
    DBLabelDistance = createDBLabelLabel(dataset, DB, id_row)

    # take k (latent features)
    k = GF.getK(len(DBLabelDistance), len(DBLabelDistance[0]))

    # featuresLatenti = list of obj, for each obj it has a list with the k values for the k latent features
    featuresLatenti = None
    if redDimID == 1:
        featuresLatenti = task6.get_PCA(DBLabelDistance, k)
    elif redDimID == 2:
        featuresLatenti = task6.get_SVD(DBLabelDistance, k)
    elif redDimID == 3:
        featuresLatenti = task6.get_LDA(DBLabelDistance, k)
    elif redDimID == 4:
        featuresLatenti = task6.get_KMeans(DBLabelDistance, k)

    # reshape featuresLatenti come as a list of features, each feature is a list of pairs (id Label, value)
    featuresLatentiShape = task7.shapeLatentFeatures(featuresLatenti, k, id_row)

    # save latent features on file
    GF.saveOnFileLatentFeatures("featuresKTask8",featuresLatentiShape,ID_space,redDimID)

# createDBLabelLabel(dataset, DB, id_row)
# dataset: dataset of images
# DB: matrix Image-Features DB[i][j]= value for feature j in img i
# id_row: Row-ID matching matrix (row in matrix code - ID in dataset)
#
# return/create DB(matrix) Label-Label
def createDBLabelLabel(dataset, DB, id_row):
    DictDBLabel = task3.getDictDatasetLabel(dataset, DB, id_row)
    DBLabel = task3.shapeDBLabel(DictDBLabel)
    DBLabelDistance = calculateLabelDistanceDB(DBLabel)

    return DBLabelDistance

# calculateLabelDistanceDB(dataset)
# dataset: matrix Label-Features
#
# calculate for each label the distance from all the other labels
def calculateLabelDistanceDB(dataset):
    res= []
    for label in tqdm(dataset):
        distanze = task2.getSimilarityVector(label,dataset)
        distanze = [j.detach().numpy().item() for (i,j) in distanze]
        res.append(distanze)
    return res
