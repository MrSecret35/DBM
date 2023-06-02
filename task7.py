import torch
import numpy as np
import csv
from tqdm import tqdm
import tensorly as tl
from tensorly.decomposition import parafac

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

    task6.

