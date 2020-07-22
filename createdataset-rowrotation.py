import numpy as np
import csv
import time
        
def loadData(file):
    ''' load the KO and WT data of one experiment'''
    tsv_file = open(file)
    read_tsv = csv.reader(tsv_file, delimiter = "\t")
    KOdata = np.array(list(read_tsv)[1:]).astype("float")
    tsv_file.close()
    return KOdata

def loadGSdata(file):
    ''' load the gold standard of one experiment and make list of tuples containing only the edges'''
    tsv_file = open(file)
    read_tsv = csv.reader(tsv_file, delimiter = "\t")
    GS = []
    for row in read_tsv:
        if int(row[2]) == 1:
            GS.append((int(row[0].strip('G')), int(row[1].strip('G'))))
    return GS

def openFiles(size, number, source, path):
    '''load all data of one experiment, depending of source Gene Net Weaver or DREAMdata'''
    if source =='GNW':
        new_path = path + '\\source_data\\GNW'
        name = '\\Yeast-' + str(number) 
        KOpath = new_path + name + '_knockouts.tsv'
        WTpath = new_path + name + '_wildtype.tsv'
        GSpath = new_path + name + '_goldstandard.tsv'

    if source == 'DREAM':
        new_path = path + '\\source_data\\DREAM4'
        name = 'insilico_size' + str(size) + '_' + str(number)
        KOpath = new_path + '\\size' + str(size) + '\\DREAM4trainingdata\\' + name  + '\\' + name + '_knockouts.tsv'
        WTpath = new_path + '\\size' + str(size) + '\\DREAM4trainingdata\\' + name  + '\\' + name + '_wildtype.tsv'
        GSpath = new_path + '\\size' + str(size) + '\\DREAM4goldstandards\\' + name + '_goldstandard.tsv'
    
    KOdata = loadData(KOpath)
    WTdata = loadData(WTpath)
    GSdata = loadGSdata(GSpath)
    
    return KOdata, WTdata, GSdata

def rotate(KOdata):
    size = len(KOdata)
    KOrotated = np.zeros((size,size))
    for column in range(size):
        for row in range(size):
            KOrotated[row-column,column] = KOdata[row,column]
            
    KOrotated = np.transpose(KOrotated)
    return KOrotated

def makeLabels(GSdata, size):
    labels = np.zeros(size*size).astype('int')
    indices = range(1,size+1)
    indices_reverse = range(size, 0, -1)
    
    edges = []
    for i in range(size):
        for j in range(size):
            source_node = indices[indices[j]-indices_reverse[i]]
            target_node = indices[j]
            edges.append((source_node, target_node))
            
    for j in range(len(edges)):
        if edges[j] in GSdata:
            labels[j] = 1
    labels = np.reshape(labels, (100,100))
    return labels

def createDataset(size, number, amount, source, path):
    '''create datasets based on variables:
            size = dimensions of the network
            number = number of the network, when multiple networks are used, this is the starting number
            amount = amount of networks to be added in dataset
            source = GNW or DREAM
            path = working directory
    '''
    completeDataset = np.array([])
    allLabels = np.array([])
    for i in range(amount):
        start_time = time.time()
        KOdata, WTdata, GSdata = openFiles(size, number, source, path)
        dataset = rotate(KOdata)
        labels = makeLabels(GSdata, size)
        completeDataset = np.append(completeDataset,dataset)
        allLabels = np.append(allLabels,labels)
        print('Loading network ' + str(number) + ' took %s seconds' % (time.time() - start_time))
        number += 1

    columns = size
    rows = amount * size
    completeDataset = completeDataset.reshape([rows,columns])
    start_time = time.time()
    name = source + '_' + str(amount) + '_' + str(size) + '_'
    np.savetxt('data\\positions\\' + name + 'data-rotated.txt', completeDataset)
    np.savetxt('data\\labels\\' + name + 'labels-rotated.txt', allLabels)
    print('Writing complete dataset took %s seconds' % (time.time() - start_time))

    return completeDataset, allLabels

path = r'C:\\Users\\Rien\\CloudDiensten\\Stack\\Documenten\\Python Scripts\\BEP'
size = 100
number = 1
amount = 10
source = 'GNW'

completeDataset, allLabels= createDataset(size, number, amount, source, path)
allLabels = np.reshape(allLabels, (size*amount, size))
print('Total amount of edges = ' + str(np.sum(allLabels)))
