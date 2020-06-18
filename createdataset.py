import csv
import numpy as np
import time

def loadData(file):
    ''' load the KO and WT data of one experiment'''
    tsv_file = open(file)
    read_tsv = csv.reader(tsv_file, delimiter = "\t")
    KOdata = np.array(list(read_tsv)[1:]).astype("float")
    tsv_file.close()
    return KOdata

def loadGSdata(file,size):
    ''' load the gold standard of one experiment and reshape it to matrix of same dimensions of KO'''
    tsv_file = open(file)
    read_tsv = csv.reader(tsv_file, delimiter = "\t")
    GSraw = list(read_tsv)
    
    GSreshaped = np.zeros([size, size])
    GS = []

    for i in range(len(GSraw)):
        GS.append([s.strip('G') for s in GSraw[i]])
    
    for i in range(len(GS)):
        x = int(GS[i][0])-1
        y = int(GS[i][1])-1
        edge = float(GS[i][2])
        GSreshaped[y][x] = edge
    
    return GSreshaped

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
    GSdata = loadGSdata(GSpath, size)
    
    return KOdata, WTdata, GSdata

def dataPrep(KOdata, WTdata, size):
    ''' create dataset of one experiment containing for every row in dataset row of gene i, column of gene i, row of gene j, column of gene j, wildtype i and wildtype j'''
    dataset = []
    for j in range(len(KOdata)):
        for i in range(len(KOdata)):
            dataset.extend(KOdata[:,i].tolist())
            dataset.extend(KOdata[i,:].tolist())
            dataset.extend(KOdata[:,j].tolist())
            dataset.extend(KOdata[j,:].tolist())
            dataset.append(WTdata[0,i].tolist())
            dataset.append(WTdata[0,j].tolist())
            
    return dataset

def reshapeLabels(GSdata):
    '''create labels array from GSdata'''
    labels = []
    for k in range(len(GSdata)):
        for l in range(len(GSdata)):
            labels.append(GSdata[l,k])
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
        print('Loading dataset: ' + str(number))
        start_time = time.time()
        KOdata, WTdata, GSdata = openFiles(size, number, source, path)
        dataset = dataPrep(KOdata, WTdata, size)
        labels = reshapeLabels(GSdata)
        completeDataset = np.append(completeDataset,dataset)
        allLabels = np.append(allLabels,labels)
        print('Processing of dataset: ' + str(number) + ' took %s seconds' % (time.time() - start_time))
        number += 1

    columns = 4 * size + 2
    rows = amount * size * size
    completeDataset = completeDataset.reshape([rows,columns])
       
    name = source + '_' + str(amount) + '_' + str(size) + '_'
    np.savetxt('data\\' + name + 'data.txt', completeDataset)
    np.savetxt('data\\' + name + 'labels.txt', allLabels)
    return completeDataset, allLabels

path = r'C:\\Users\\Rien\\CloudDiensten\\Stack\\Documenten\\Python Scripts\\BEP'
size = 100
number = 1
amount = 5
source = 'GNW'

completeDataset, allLabels = createDataset(size, number, amount, source, path)