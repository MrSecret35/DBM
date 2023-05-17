import csv
import numpy
import task2
def start(Caltech101, ReteNeurale):
    # prendere etichetta
    N_etichetta = int(input("inserisci l'etichetta (tra 0 e 94): "))
    # suddividere il DB in etichette
    ID_space = ''
    while ID_space != '1' and ID_space != '2' and ID_space != '3':
        ID_space = input("select space: \n 1 - layer3\n 2 - avg\n 3 - last\n")
        if (ID_space != '1' and ID_space != '2' and ID_space != '3'):
            print("insert a valid selection")

    dataset = getDatasetLabel(Caltech101,ID_space)

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

    print("etichetta: ", N_etichetta, "quella più simile: ", int(simList[1][0]/2))
    print("classe: ",
          Caltech101.annotation_categories[N_etichetta],
          "quella più simile: ",
          Caltech101.annotation_categories[int(simList[1][0]/2)])

def getDatasetLabel(Caltech101,ID_space):
    fileVector = {}
    if ID_space == '1':
        fileVector = open('Data\dataVectorLayer3.csv', 'r')
    elif ID_space == '2':
        fileVector = open('Data\dataVectorAVGPool.csv', 'r')
    elif ID_space == '3':
        fileVector = open('Data\dataVectorLast.csv', 'r')

    readerVector = csv.reader(fileVector, delimiter=';', quoting=csv.QUOTE_NONNUMERIC)
    DB = numpy.array(list(readerVector))

    res= {}
    for i in range(len(DB)):
        img, label = Caltech101[i*2]
        if label not in res.keys():
            res[label]= []

        res[label].append(DB[i])

    return res

