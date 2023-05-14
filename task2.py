import csv
import numpy
import matplotlib.pyplot as plt
import genericFunction as GF
import math
from tqdm import tqdm

from task1 import Task1


def start(Caltech101, ReteNeurale):
    task1 = Task1()

    #TODO: fare i controlli per gli ID
    #ask what image
    ID_img = int(input("insert an ID for an Image (odd number):"))

    ID_space = input("select space: \n 1 - layer3\n 2 - avg\n 3 - last\n")


    fileVector ={}
    if ID_space == '1':
        fileVector = open('Data\dataVectorLayer3.csv', 'r')
    elif ID_space == '2':
        fileVector = open('Data\dataVectorAVGPool.csv', 'r')
    elif ID_space == '3':
        fileVector = open('Data\dataVectorLast.csv', 'r')


    readerVector = csv.reader(fileVector, delimiter=';')
    DB = numpy.array(list(readerVector))

    img, label = Caltech101[ID_img]


    [vector_last, vector_avgpool, vector_layer3, class_id] = task1.Extractor(ID_img, Caltech101, ReteNeurale)


    # non indipendenti (vanno chiamati in ordine)
    #tensor = GF.IMGtoTensor(img)
    #vector_last = task1.getFC(tensor, ReteNeurale)
    #vector_avgpool = task1.getAvgpoolFlatten('avgpool')
    #vector_layer3 = task1.getLayer3Flatten('layer3')

    listaSim = []
    for i in tqdm(range(len(DB))):
        d = -1
        if ID_space == '1':
            d = distanza(vector_layer3.detach().numpy(), DB[i])
        elif ID_space == '2':
            d = distanza(vector_avgpool.detach().numpy(), DB[i])
        elif ID_space == '3':
            d = distanza(vector_last.detach().numpy(), DB[i])

        listaSim.append((i*2,d))

    #TODO: trovare gli n simili (non solo 1)
    min= listaSim[0]
    for i in range(len(DB)):
        if listaSim[i][1] < min[1]:
            min = listaSim[i]

    print("ID immagine piú simile: " , min[0])
    imgRes, labelRes = Caltech101[min[0]]
    #plt.imshow(imgRes)
    #plt.show()

    f, axarr = plt.subplots(2, 1)
    axarr[0].imshow(img)
    axarr[0].set_title("immagine scelta")
    axarr[1].imshow(imgRes)
    axarr[1].set_title("immagine più simile nel DB")
    plt.show()

def distanza(v1, v2):
    if len(v1) != len(v2) :
        print("error")
        return -1

    res = 0
    for i in range(len(v1)):
        res += (v1[i].item()-float(v2[i].item()))**2

    return math.sqrt(res)

