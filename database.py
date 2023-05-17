import csv
import numpy

# IDSpace()
# takes as input an ID that identifies the vector space of the features
def IDSpace():
    ID_space = ''
    while ID_space != '1' and ID_space != '2' and ID_space != '3':
        ID_space = input("select space: \n 1 - layer3\n 2 - avg\n 3 - last\n")
        if (ID_space != '1' and ID_space != '2' and ID_space != '3'):
            print("insert a valid selection")
    return int(ID_space)

# getDB(ID_space)
# ID_space: id of the vector space of the features to collect the data
# reads the desired vector space from file and returns it as a matrix
def getDB(ID_space):
    fileVector = {}
    if ID_space == 1:
        fileVector = open('Data\dataVectorLayer3.csv', 'r')
    elif ID_space == 2:
        fileVector = open('Data\dataVectorAVGPool.csv', 'r')
    elif ID_space == 3:
        fileVector = open('Data\dataVectorLast.csv', 'r')

    readerVector = csv.reader(fileVector, delimiter=';', quoting=csv.QUOTE_NONNUMERIC)
    DB = numpy.array(list(readerVector))
    return DB