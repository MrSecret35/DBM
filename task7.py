import torch
import numpy as np
import csv
from tqdm import tqdm
import tensorly as tl
from tensorly.decomposition import parafac

import genericFunction as GF
import database as DBFunc
#


def processing(dataset,ReteNeurale):

    # take DB
    ID_space = DBFunc.IDSpace()
    DB = DBFunc.getDB(ID_space)
    id_row = DBFunc.getDBID()

    # take k (latent features)
    k = int(input("inserisci k (features latenti): "))

    DB_tensor=tl.tensor(DB)
    # take k latent feature with Parafac decomposition
    (weights, factors)= parafac(DB_tensor,rank=k)

    # reshape featuresLatenti come as a list of features, each feature is a list of pairs (ID label, value)
    featuresLatentiShape = shapeLatentFeatures(factors[0], k, id_row,dataset)

    # save latent features on file
    GF.saveOnFileLatentFeatures("featuresKTask7", featuresLatentiShape, ID_space, 5)

# shapeLatentFeatures(featuresLatenti,k,id_row,dataset)
# featuresLatenti: list of obj, for each obj it has a list with the k values for the k latent features
# k: number of latent features
# id_row: Row-ID matching matrix (row in matrix code - ID in dataset)
#
# reshape featuresLatenti come as a list of features, each feature is a list of pairs (label img, value), each feature is sorted in descending order
def shapeLatentFeatures(featuresLatenti,k,id_row,dataset):
    featuresLatentiShape = []
    for i in tqdm(range(k)):
        d = []
        for j in range(len(featuresLatenti)):
            id=DBFunc.getIDfromRow(j+1,id_row)
            img,label = dataset[id]
            d.append((label, featuresLatenti[j][i]))
        d = sorted(d, key=lambda tup: tup[1], reverse=True)
        featuresLatentiShape.append(d)

    return featuresLatentiShape