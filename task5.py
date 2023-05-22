from tqdm import tqdm

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
    print(listArchi)
    # calcolare MT matrice transizione
    # calcolare matrice di teletrasporto
    # sommare le matrici
    # trovare autovettore con autovalore 1
    # prendere le m immagini più significative

def getGraph(dataset, n):
    listVertici=[]
    listArchi=[]

    tqdm.tqdm = False
    for i in range(len(dataset)):
        listVertici.append(DBFunc.getIDfromRow(i+1))

        simList = task2.getSimilarityVector(dataset[i], dataset)
        simList = sorted(simList, key=lambda tup: tup[1])
        simList = simList[0:n]

        for j in range(len(simList)):
            listArchi.append( (DBFunc.getIDfromRow(i+1), DBFunc.getIDfromRow(simList[j][0]) ) )

    return listVertici, listArchi


