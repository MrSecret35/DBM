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
import task5
import task6

def processing(dataset,ReteNeurale):
    Beta=0.5

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

    # take ID label
    N_etichetta = int(input("inserisci l'etichetta (tra 0 e 94): "))

    # take 'n' range for graph
    n = int(input("inserisci n (numero immagini più vicine): "))

    # take number of images to print
    m = int(input("insert n (numero immagini da stampare):"))

    # calculate the graph
    print("Prendo la lista Vertici e la lista Archi")
    listVertici,listArchi = task5.getGraph(DB,n,DBDistanceLatent, id_row)

    # calculate MT == transition matrix
    print("Calcolo la matrice di Transizione MT")
    M = task5.fillTMatrix(DBDistanceLatent, listArchi, n, Beta, id_row)

    # calculate G == teleportation matrix
    # G = teletrasporto
    print("Calcolo la matrice di teletrasporto G")
    G = task5.fillGMatrix(dataset, DB, Beta, N_etichetta, id_row)

    # calculate Z == sum between M and G
    print("Calcolo la nuova matrice di transizione Z")
    Z = task5.getZMatrix(Beta, M, G)

    # take autovector
    print("Calcolo l'autovettore")
    V = task5.getAutovector(Z)

    # take m ID from V
    print("Stampo")
    m_id = task5.takeMID(V, m, id_row)

    # print m images
    GF.printNIMG(m_id, dataset)