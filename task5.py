import numpy
from tqdm import tqdm
from collections import Counter

import database as DBFunc
import genericFunction as GF
import task2
import task3

def processing(dataset,ReteNeurale):
    Beta= 0.5
    id_row= DBFunc.getDBID()

    # take DB
    ID_space = DBFunc.IDSpace()
    DB = DBFunc.getDB(ID_space)
    datasetSim = DBFunc.getDistanceDB(1, ID_space)

    # take ID label
    N_etichetta = int(input("inserisci l'etichetta (tra 0 e 94): "))

    # take 'n' range for graph
    n = int(input("inserisci n (numero immagini più vicine): "))

    # take number of images to print
    m = int(input("insert n (numero immagini da stampare):"))


    # calculate the graph
    print("Prendo la lista Vertici e la lista Archi")
    listVertici,listArchi = getGraph(DB,n,datasetSim, id_row)

    # calculate MT == transition matrix
    # id -- getIDfromRow( riga+1 )
    # in un range le righe/colonne partono da 0 mentre le righe nel DB partono da 1
    print("Calcolo la matrice di Transizione MT")
    M = fillTMatrix(datasetSim, listArchi, n, Beta, id_row)

    # calculate G == teleportation matrix
    #G = teletrasporto
    print("Calcolo la matrice di teletrasporto G")
    G = fillGMatrix(dataset, DB,Beta, N_etichetta,id_row)

    # calculate Z == sum between M and G
    print("Calcolo la nuova matrice di transizione Z")
    Z = getZMatrix(Beta,M,G)

    #take autovector
    print("Calcolo l'autovettore")
    V = getAutovector(Z)

    # take m ID from V
    print("Stampo")
    m_id = takeMID(V,m,id_row)

    #print m images
    GF.printNIMG(m_id,dataset)

# getGraph(dataset, n, datasetSim,id_row)
# dataset: dataset of images
# n: range of img for calculating arcs
# datasetSim: img-img matrix, contains for each img a list of distances with all the other imgs
# id_row: Row-ID matching matrix (row in matrix code - ID in dataset)
#
# each img becomes a vertex and calculates for each img n arcs towards the n most similar imgs
# listVertici: list of vertices (1 per img)
# listArchi: arc list, tuples (IDimg, IDimg), n arcs for each node
def getGraph(dataset, n, datasetSim,id_row):
    listVertici=[]
    listArchi=[]

    for i in tqdm(range(len(datasetSim))):
        listVertici.append(DBFunc.getIDfromRow(i+1,id_row))
        listSim = zip([z for z in range(len(datasetSim[i]))] , datasetSim[i])
        listSim = sorted(listSim, key=lambda tup: tup[1])
        for j in range(n):
            #listArchi.append( (DBFunc.getIDfromRow(i+1,id_row), DBFunc.getIDfromRow(listSim[j+1][0],id_row) ))
            listArchi.append((i, listSim[j+1][0]))
    return listVertici, listArchi

# getMatrix(N)
# N: integer
#
# return/generates an NxN matrix od 0s
def getMatrix(N):
    M = []
    for _ in range(N):
        M.append([0 for _ in range(N)])

    return M

# fillTMatrix(datasetSim, listArchi, n, Beta,id_row)
# datasetSim: img-img matrix, contains for each img a list of distances with all the other imgs
# listArchi: arc list, tuples (IDimg, IDimg)
# n: number of arcs for each node
# Beta
# id_row: Row-ID matching matrix (row in matrix code - ID in dataset)
#
# for each arc (i,j) fills the corresponding field in a matrix M with Beta/n
# returns the matrix M
def fillTMatrix(datasetSim, listArchi, n, Beta,id_row):
    value = Beta/n
    M=getMatrix(len(datasetSim))

    for (i,j) in tqdm(listArchi):
        #row_i = DBFunc.getRowfromID(i,id_row)
        #row_j = DBFunc.getRowfromID(j,id_row)
        M[i][j]= value
    return M

#crea e riempie la matrice MT
def fillTMatrix_old(datasetSim, listArchi, n, Beta,id_row):
    value = 1 / n * Beta
    M=[]
    for i in tqdm(range(len(datasetSim))):
        d=[]
        d= [value if (i,j) in listArchi else 0 for j in range(len(datasetSim[i]))]
        for j in range(len(datasetSim[i])):
            ID_i = DBFunc.getIDfromRow(i+1,id_row)
            ID_j = DBFunc.getIDfromRow(j+1,id_row)
            if (ID_i,ID_j) in listArchi:
                d.append(value)
            else:
                d.append(0)

        M.append(d)

    return M

#fillGMatrix(dataset, DB, Beta, N_etichetta,id_row)
# dataset: dataset of images
# DB: Database/matrix Image-Features
# Beta
# N_etichetta: ID of Label
# id_row: Row-ID matching matrix (row in matrix code - ID in dataset)
#
# for each img in the Label it fills the field in the matrix G with ( 1-Beta / Nimg in the label )
# returns the matrix G
def fillGMatrix(dataset, DB, Beta, N_etichetta,id_row):
    #prendere img dell'etichetta
    DBLabel = task3.getDictDatasetLabel(dataset,DB,id_row)
    imgLabel= DBLabel[N_etichetta]
    print("numero immagini nella label: ", len(imgLabel))
    value = (1-Beta)/len(imgLabel)

    d = []
    for j in range(len(DB)):
        if any(numpy.array_equal(DB[j], x) for x in imgLabel):
            d.append(value)
        else:
            d.append(0)

    G = []
    for i in range(len(DB)):
        G.append(d)

    return G

#crea un dizionario che ha l'id dell'arco e quante volte questo compare come primo elemento della tupla 
def getOuterEdges(listArchi):
     c = Counter(el[0] for el in listArchi)

# getZMatrix(Beta,M,G)
# Beta
# M: matrix NxN
# G: matrix NxN
#
# sum of 2 matrices
def getZMatrix(Beta,M,G):
    Z = []

    for i in tqdm(range(len(M))):
        d=[]
        for j in range(len(M[i])):
            x = M[i][j] + G[i][j]
            d.append(x)
        Z.append(d)
    return Z

# getAutovector(Z)
# Z: matrix NxN
#
# return/find the eigenvector with maximum eigenvalue
def getAutovector(Z):
    res = []
    Z= numpy.array(Z)
    eigenvalues, eigenvectors = numpy.linalg.eig(Z.T)

    indexArgMin1 = numpy.argmax(eigenvalues)

    res=eigenvectors[:,indexArgMin1]
    res = res.real
    res = res / res.sum()

    return res

# takeMID(V, m, id_row)
# V: array of number
# m: integer
# id_row: Row-ID matching matrix (row in matrix code - ID in dataset)
#
# takes the first m values in V (descending order)
def takeMID(V, m, id_row):
    V = zip([z + 1 for z in range(len(V))], V)
    V = sorted(V, key=lambda tup: tup[1], reverse=True)
    m_id = [ i for (i,j) in V[0:m] ]
    m_id = [DBFunc.getIDfromRow(x, id_row) for x in m_id]
    return m_id