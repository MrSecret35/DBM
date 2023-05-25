import numpy
from tqdm import tqdm
from collections import Counter
import database as DBFunc
import task2
import task3

def processing(Caltech101,ReteNeurale):
    Beta= 0.5
    id_row= DBFunc.getDBID()
    # prendere DB
    ID_space = DBFunc.IDSpace()
    # prendere etichetta
    N_etichetta = int(input("inserisci l'etichetta (tra 0 e 94): "))
    # prendere n (immagini più vicine)
    n = int(input("inserisci n (numero immagini più vicine): "))


    # take DB
    DB = DBFunc.getDB(ID_space)
    # calcolare grafo
    datasetSim = DBFunc.getDistanceDB(1, ID_space)

    print("Prendo la lista Vertici e la lista Archi")
    listVertici,listArchi = getGraph(DB,n,datasetSim,id_row)

    # calcolare MT matrice transizione
    # per ottenere un id getIDfromRow( riga+1 )
    # in un range le righe/colonne partono da 0 mentre le righe nel DB partono da 1
    print("Calcolo la matrice di Transizione MT")
    #M = getMatrix(len(DB))
    M = fillTMatrix(datasetSim, listArchi, n, Beta,id_row)

    # calcolare matrice di teletrasporto
    #G = teletrasporto
    print("Calcolo la matrice di teletrasporto G")
    G = fillGMatrix(Caltech101, DB,Beta, N_etichetta,id_row)

    # sommare le matrici
    print("Calcolo la nuova matrice di transizione Z")
    Z = getZMatrix(Beta,M,G)


    # trovare autovettore con autovalore 1

    # prendere le m immagini più significative

def getGraph(dataset, n, datasetSim,id_row):
    listVertici=[]
    listArchi=[]

    for i in tqdm(range(len(datasetSim))):
        listVertici.append(DBFunc.getIDfromRow(i+1,id_row))
        #listSim = zip([z+1 for z in range(n+2)], datasetSim[i][0:(n+2)])
        listSim = zip([z+1 for z in range(len(datasetSim[i]))] , datasetSim[i])
        listSim = sorted(listSim, key=lambda tup: tup[1])
        for j in range(n):
            listArchi.append( (DBFunc.getIDfromRow(i+1,id_row), DBFunc.getIDfromRow(listSim[j+1][0],id_row) ))
    return listVertici, listArchi

#crea la matrice M con liste vuote 
def getMatrix(N):
    M = []
    for _ in range(N):
        M.append([0 for _ in range(N)])

    return M

def fillTMatrix(datasetSim, listArchi, n, Beta,id_row):
    value = 1 / n * Beta
    M=getMatrix(len(datasetSim))

    for (i,j) in tqdm(listArchi):
        row_i = DBFunc.getRowfromID(i,id_row)
        row_j = DBFunc.getRowfromID(j,id_row)
        M[row_i-1][row_j-1]= value
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
        
def fillGMatrix(Caltech101, DB,Beta, N_etichetta,id_row):
    #prendere img dell'etichetta
    DBLabel = task3.getDatasetLabel(Caltech101,DB)
    imgLabel= DBLabel[N_etichetta]
    value = 1/len(imgLabel) * (1-Beta)

    d = []
    for j in tqdm(range(len(DB))):
        if any(numpy.array_equal(DB[j], x) for x in imgLabel):
            d.append(value)
        else:
            d.append(0)

    G = []
    for i in tqdm(range(len(DB))):
        G.append(d)

    return G

#crea un dizionario che ha l'id dell'arco e quante volte questo compare come primo elemento della tupla 
def getOuterEdges(listArchi):
     c = Counter(el[0] for el in listArchi)

def getZMatrix(Beta,M,G):
     Z = Z = [ M[i]+G[i] for i in range(len(M))]
     return Z
