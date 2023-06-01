import torch
import numpy as np
import csv
from tqdm import tqdm
from tensorly.decomposition import parafac

import database as DBFunc

#


def processing(Caltech101,ReteNeurale):
    # prendere DB
    ID_space = DBFunc.IDSpace()
    id_row = DBFunc.getDBID()

    # prendere k (features latenti)
    k = int(input("inserisci k (features latenti): "))

    # take DB
    DB = DBFunc.getDB(ID_space)
    #DB_tensor = torch.tensor(DB)
    DB_tensor= np.array([ np.array([ np.array([DB[i][j]],dtype='uint8') for j in range(len(DB[i]))]) for i in range(len(DB))])

    #dec = parafac(vector_last.unsqueeze(-1).detach().numpy(), rank=3)
    (weights, factors)= parafac(DB,rank=4)

    print(weights)
    print(factors)
