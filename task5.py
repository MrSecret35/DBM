from tqdm import tqdm
from collections import Counter
import database as DBFunc
import task2

def processing(Caltech101,ReteNeurale):
    print("ciao")

    # prendere DB
    ID_space = DBFunc.IDSpace()
    # prendere etichetta
    N_etichetta = int(input("inserisci l'etichetta (tra 0 e 94): "))
    # prendere n (immagini più vicine)
    n = int(input("inserisci n (numero immagini più vicine): "))


    # take DB
    DB = DBFunc.getDB(ID_space)
    # calcolare grafo
    listVertici,listArchi = getGraph(DB,n)
    print(listVertici)
    print(listArchi[0])
    # calcolare MT matrice transizione
    M = getMatrix(len(DB))
    print(M.shape)
    # calcolare matrice di teletrasporto
    # sommare le matrici
 
    # trovare autovettore con autovalore 1
    # prendere le m immagini più significative

def getGraph(dataset, n):
    listVertici=[]
    listArchi=[]

    for i in range(len(dataset)):
        listVertici.append(DBFunc.getIDfromRow(i+1))

        simList = task2.getSimilarityVector(dataset[i], dataset)
        simList = sorted(simList, key=lambda tup: tup[1])
        simList = simList[0:n]

        for j in range(len(simList)):
            listArchi.append( (DBFunc.getIDfromRow(i+1), DBFunc.getIDfromRow(simList[j][0]) ) )

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

#riempie la matrice M 
def fillMMatrix(M,N, listArchi, counter):
    for i in range(N): 
        for j in range(N):
            if (i,j) in listArchi or (j,i) in listArchi:
                  M[i][j] = counter[i]
            else:
               M[i][j] = 0
        

#crea un dizionario che ha l'id dell'arco e quante volte questo compare come primo elemento della tupla 
def getOuterEdges(listArchi):
     c = Counter(el[0] for el in listArchi)

def getZMatrix(Beta,N,M):
     tmp = getMatrix(N,False)
     Z = (1-Beta)*tmp+Beta*M
     return Z
