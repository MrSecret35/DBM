from tqdm import tqdm
from collections import Counter
import database as DBFunc
import task2

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
    listVertici,listArchi = getGraph(DB,n,datasetSim,id_row)

    # calcolare MT matrice transizione
    #M = getMatrix(len(DB))
    M = fillTMatrix(datasetSim, listArchi, n, Beta,id_row)
    print(M)
    # calcolare matrice di teletrasporto
    # sommare le matrici
 
    # trovare autovettore con autovalore 1
    # prendere le m immagini più significative

def getGraph(dataset, n, datasetSim,id_row):
    listVertici=[]
    listArchi=[]

    for i in range(len(datasetSim)):
        listVertici.append(DBFunc.getIDfromRow(i+1,id_row))
        listSim = zip([z+1 for z in range(len(datasetSim[i]))] , datasetSim[i])
        listSim = sorted(listSim, key=lambda tup: tup[1])
        for j in range(n):
            listArchi.append( (DBFunc.getIDfromRow(i+1,id_row), DBFunc.getIDfromRow(listSim[j][0],id_row) ))
    return listVertici, listArchi

#crea la matrice M con liste vuote 
def getMatrix(N,empty):
       M = []
       if empty == True:
        for _ in range(N):
            M.append([])
       else: 
         for _ in range(N):
            M.append([1/N])

       return M

#crea e riempie la matrice MT
def fillTMatrix(datasetSim, listArchi, n, Beta,id_row):
    value = 1 / n * Beta
    M=[]
    for i in tqdm(range(len(datasetSim))):
        d=[]
        for j in range(len(datasetSim[i])):
            ID_i = DBFunc.getIDfromRow(i+1,id_row)
            ID_j = DBFunc.getIDfromRow(j+1,id_row)
            if (ID_i,ID_j) in listArchi:
                d.append(value)
            else:
                d.append(0)
        M.append(d)

    return M
        

#crea un dizionario che ha l'id dell'arco e quante volte questo compare come primo elemento della tupla 
def getOuterEdges(listArchi):
     c = Counter(el[0] for el in listArchi)

def getZMatrix(Beta,N,M):
     tmp = getMatrix(N,False)
     Z = (1-Beta)*tmp+Beta*M
     return Z
