import torch
import numpy as np
import csv
from tqdm import tqdm
import tensorly as tl
from tensorly.decomposition import parafac

import genericFunction as GF
import database as DBFunc
import task6
#


def processing(Caltech101,ReteNeurale):
    # prendere DB
    ID_space = DBFunc.IDSpace()
    id_row = DBFunc.getDBID()

    # prendere k (features latenti)
    k = int(input("inserisci k (features latenti): "))

    # take DB
    DB = DBFunc.getDB(ID_space)
    DB_tensor=tl.tensor(DB)

    (weights, factors)= parafac(DB_tensor,rank=k)

    featuresLatentiShape = task6.shapeLatentFeatures(factors[0], k, id_row)

    GF.saveOnFileLatentFeatures("featuresKTask7", featuresLatentiShape, ID_space, 5)

