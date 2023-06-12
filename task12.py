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

    DBDistanceLatent = DBFunc.getLatentDistanceDB(ID_Task, ID_space, ID_Dec)