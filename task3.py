import csv
import numpy

import database as DBFunc
import task2
def start(Caltech101, ReteNeurale):
    # prendere etichetta
    N_etichetta = int(input("inserisci l'etichetta (tra 0 e 94): "))
    # suddividere il DB in etichette
    ID_space = DBFunc.IDSpace()

    DB = DBFunc.getDB(ID_space)
    dataset = getDatasetLabel(Caltech101,DB)

    # calcolare il leader per ogni etichetta
    for i in dataset:
        vectorLeader = dataset[i][0]
        for img in dataset[i]:
            vectorLeader += img
            vectorLeader = [x/2 for x in vectorLeader]
        dataset[i]= vectorLeader

    # calcolare i più simili alla nostra etichetta
    simList = task2.getSimilarityVector(dataset[N_etichetta],dataset)
    simList = sorted(simList, key=lambda tup: tup[1])

    #stampa
    print("etichetta: ", N_etichetta, "quella più simile: ", int(simList[1][0]/2))
    print("classe: ",
          Caltech101.annotation_categories[N_etichetta],
          "quella più simile: ",
          Caltech101.annotation_categories[int(simList[1][0]/2)])

def getDatasetLabel(Caltech101,DB):

    res= {}
    for i in range(len(DB)):
        img, label = Caltech101[i*2]
        if label not in res.keys():
            res[label]= []

        res[label].append(DB[i])

    return res

