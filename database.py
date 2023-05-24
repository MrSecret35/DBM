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

def getDistanceDB(ID_Distance,ID_Space):
    fileDistance = {}
    if ID_Distance == 1:
        if ID_Space == 1:
            fileDistance = open('Distance\EuclideanDistance_Layer3.csv', 'r')
        elif ID_Space == 2:
            fileDistance = open('Distance\EuclideanDistance_AVGPool.csv', 'r')
        elif ID_Space == 3:
            fileDistance = open('Distance\EuclideanDistance_Last.csv', 'r')

        readerDB = csv.reader(fileDistance, delimiter=';', quoting=csv.QUOTE_NONNUMERIC)
        DB_distance = numpy.array(list(readerDB))

        return DB_distance

def getDBID():
    file = open('Data\IDtoRow.csv', 'r')
    reader = csv.reader(file, delimiter=';', quoting=csv.QUOTE_NONNUMERIC)
    return numpy.array(list(reader))

def getIDfromRow(row):
    id_row = getDBID()
    res = [t for t in id_row if t[1]==row]
    return int(res[0][0])

def getIDfromRow(row,id_row):
    res = [t for t in id_row if t[1]==row]
    return int(res[0][0])