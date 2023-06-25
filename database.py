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

# IDTaskIMG()
# takes as input an ID that identifies the task to use between 6 / 9
def IDTaskIMG():
    ID_task = ''
    while ID_task != '6' and ID_task != '9':
        ID_task = input("select space: \n 6 - task6\n 9 - task9\n")
        if (ID_task != '6' and ID_task != '9'):
            print("insert a valid selection")
    return int(ID_task)

# IDTaskLabel()
# takes as input an ID that identifies the task to use between 7 / 8
def IDTaskLabel():
    ID_task = ''
    while ID_task != '7' and ID_task != '8':
        ID_task = input("select space: \n 7 - task7\n 8 - task8\n")
        if (ID_task != '7' and ID_task != '8'):
            print("insert a valid selection")
    return int(ID_task)

# IDTask()
# takes as input an ID that identifies the task to use between 6 / 7 / 8 / 9
def IDTask():
    ID_task = ''
    while ID_task != '6' and ID_task != '7' and ID_task != '8' and ID_task != '9':
        ID_task = input("select space: \n 6 - task6\n 7 - task7\n 8 - task8\n 9 - task9\n")
        if (ID_task != '6' and ID_task != '7' and ID_task != '8' and ID_task != '9'):
            print("insert a valid selection")
    return int(ID_task)

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

# getDistanceDB(ID_Distance,ID_Space)
# ID_Distance: id of the distance method
# ID_space: id of the vector space of the features to collect the data
# reads the desired DB (with obj-obj distance) from file and returns it as a matrix
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

# getLatentDB(ID_Task,ID_space,ID_Dec)
# ID_Task: id of the vector space of the features to collect the data
# ID_space: id of the vector space of the features to collect the data
# ID_Dec: id of the decomposition method
# reads the desired DB (created with latent features) from file and returns it as a matrix
def getLatentDB(ID_Task,ID_space,ID_Dec):
    fileDir="LatentFeatures/"
    if ID_Task == 6:
        fileDir+="featuresKTask6"
    elif ID_Task == 7:
        fileDir+="featuresKTask7"
    elif ID_Task == 8:
        fileDir+="featuresKTask8"
    elif ID_Task == 9:
        fileDir+="featuresKTask9"

    if ID_space == 1:
        fileDir += "_Layer3"
    elif ID_space == 2:
        fileDir += "_AVGPool"
    elif ID_space == 3:
        fileDir += "_VectorLast"

    if ID_Dec == 1:
        fileDir+="_PCA"
    elif ID_Dec == 2:
        fileDir+="_SVD"
    elif ID_Dec == 3:
        fileDir += "_LDA"
    elif ID_Dec == 4:
        fileDir += "_KMeans"
    elif ID_Dec == 5:
        fileDir += "_CP"

    fileDir += ".csv"
    file = open(fileDir, 'r')

    readerDB = csv.reader(file, delimiter=';')
    DB= [[eval(elem) for elem in line] for line in readerDB ]

    return DB

# getLatentDistanceDB(ID_Task,ID_space,ID_Dec)
# ID_Task: id of the vector space of the features to collect the data
# ID_space: id of the vector space of the features to collect the data
# ID_Dec: id of the decomposition method
# reads the desired DB (with obj-obj distance (created with latent features)) from file and returns it as a matrix
def getLatentDistanceDB(ID_Task,ID_space,ID_Dec):
    fileDir="Distance/EuclideanDistance_"
    if ID_Task == 6:
        fileDir+="featuresKTask6"
    elif ID_Task == 9:
        fileDir+="featuresKTask9"

    if ID_space == 1:
        fileDir += "_Layer3"
    elif ID_space == 2:
        fileDir += "_AVGPool"
    elif ID_space == 3:
        fileDir += "_VectorLast"

    if ID_Dec == 1:
        fileDir+="_PCA"
    elif ID_Dec == 2:
        fileDir+="_SVD"
    elif ID_Dec == 3:
        fileDir += "_LDA"
    elif ID_Dec == 4:
        fileDir += "_KMeans"

    fileDir += ".csv"
    file = open(fileDir, 'r')

    readerDB = csv.reader(file, delimiter=';', quoting=csv.QUOTE_NONNUMERIC)
    DB_distance = numpy.array(list(readerDB))

    return DB_distance

# getDBID():
#return IDtoRow file (relates the rows of a csv table (1,2,3,...) to the IDs of the dataset objs)
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

def getRowfromID(id,id_row):
    res = [t for t in id_row if t[0]==id]
    return int(res[0][1])