import csv
import numpy as np
import time
from scipy import stats

def loadData(file):
    ''' load the KO, KD or WT data of one experiment'''
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
        GSreshaped[x][y] = edge
    
    return GSreshaped

def openFiles(size, number, source, path):
    '''load all data of one experiment, depending of source Gene Net Weaver or DREAMdata'''
    if source[0] =='GNW':
        new_path = path + '\source_data\GNW\\' + '\size' + str(size) + '\\' + source[1] + '\\' 
        name = '\\Yeast-' + str(number) 
        KOpath = new_path + source[2] + '\\' + name + '_knockouts.tsv'
        KDpath = new_path + source[2] + '\\' + name + '_knockdowns.tsv'
        WTpath = new_path + source[2] + '\\' + name + '_wildtype.tsv'
        GSpath = new_path + 'goldstandard\\' + name + '_goldstandard.tsv'
    elif source[0] == 'DREAM':
        new_path = path + '\\source_data\\DREAM4' + '\\size' + str(size)
        name = 'insilico_size' + str(size) + '_' + str(number)
        KOpath = new_path + '\\DREAM4trainingdata\\' + name  + '\\' + name + '_knockouts.tsv'
        KDpath = new_path + '\\DREAM4trainingdata\\' + name  + '\\' + name + '_knockdowns.tsv'
        WTpath = new_path + '\\DREAM4trainingdata\\' + name  + '\\' + name + '_wildtype.tsv'
        GSpath = new_path + '\\DREAM4goldstandards\\' + name + '_goldstandard.tsv'
    elif source[0] == 'RANDOM':
        new_path = path + '\\source_data\\RANDOM'
        name = '\\random_network_' + str(number) 
        KOpath = new_path + name + '_knockouts.tsv'
        KDpath = new_path + name + '_knockdowns.tsv'
        GSpath = new_path + '\goldstandard\\' + name + '_goldstandard.tsv'
    
    KO = loadData(KOpath)
    KD = loadData(KDpath)
    WT = loadData(WTpath)
    GS = loadGSdata(GSpath, size)
    
    return KO, KD, WT, GS

def dataPrep(KO, KD, WT, GS, preprocessing):
    ''' create dataset of one experiment containing for every row in dataset row of gene i, column of gene i, row of gene j, column of gene j, wildtype i and wildtype j'''
    dataset = []
    # set diagonal to WT
    for i in range(len(KO)):
        KO[i,i] = WT[0][i]
    logKO = (np.ma.log(KO)).filled(0)
    logKD = (np.ma.log(KD)).filled(0)
    Zko = np.absolute(stats.zscore(logKO, axis = 0, ddof = 0))
    Zkd = np.absolute(stats.zscore(logKD, axis = 0, ddof = 0))
    Zko_nan_index = np.isnan(Zko)
    Zkd_nan_index = np.isnan(Zkd)
    Zko[Zko_nan_index] = 0
    Zkd[Zkd_nan_index] = 0
    Zmax = np.maximum(Zko, Zkd)    # use mean or max 

    # rotate the complete matrix in order to train on the first column every time
    for j in range(len(KO)):
        if preprocessing == 'KO':
            rotated = np.roll(KO, -j, axis = 0)
        elif preprocessing == 'logKO':
            rotated = np.roll(logKO, -j, axis = 0)
        elif preprocessing == 'Zmax':
            rotated = np.roll(Zmax, -j, axis = 0)
        for i in range(len(KO)):
            dataset.extend(rotated[:,i].tolist())
            
    return dataset, KO, rotated#, logKO, Zko, Zmax

def reshapeLabels(GS, size):
    '''create labels array from GS'''
    labels = np.reshape(GS,[size*size,1])
    return labels, GS

def createDataset(size, number, amount, source, path, preprocessing):
    '''create dataset based on variables:
            size = dimensions of the network
            number = number of the network, when multiple networks are used, this is the starting number
            amount = amount of networks to be added in dataset
            source = GNW or DREAM
            path = working directory
            preprocessing = raw data, logarithmic data or Z stats
    '''
    if source[0] == 'DREAM':
        savename = 'data\\' + source[0] + '_' + str(number) + '_' + str(amount) + '_' + str(size) + '_' + preprocessing
    elif source[0] =='GNW':
        savename = 'data\\' + source[0] + '-' + source[1] + '-' + source[2] + '_' + str(amount) + '_' + str(size) + '_' + preprocessing
    elif source[0] =='RANDOM':
        savename = 'data\\' + source[0] + '_' + str(amount) + '_' + str(size) + '_' + preprocessing
    completeDataset = np.array([])
    allLabels = np.array([])
    for i in range(amount):
        start_time = time.time()
        KO, KD, WT, GS = openFiles(size, number, source, path)
        dataset, KO = dataPrep(KO, KD, WT, GS, preprocessing)
        labels, GS = reshapeLabels(GS, size)
        completeDataset = np.append(completeDataset,dataset)
        allLabels = np.append(allLabels,labels)
        print('Loading network ' + str(number) + ' took %s seconds' % (time.time() - start_time))
        number += 1

    columns = 1 * size # + 2
    rows = amount * size * size
    completeDataset = completeDataset.reshape([rows,columns])
    start_time = time.time()
    np.savetxt(savename + '-data.txt', completeDataset)
    np.savetxt(savename + '-labels.txt', allLabels, fmt='%d')
    print('Writing complete dataset took %s seconds' % (time.time() - start_time))

    return completeDataset, allLabels, GS, KO

def create_all():
    path = r'C:\Users\Rien\CloudDiensten\Stack\Documenten\Python Scripts\BEP'
    size = 100
    source_dream = ('DREAM', '~', '~')
    source_gnw = ('GNW', 'greedy', 'nonoise')
    preprocessing = 'KO' #KO, logKO, Zmax
    # dreamfiles
    for i in range(5):
        amount = 1
        number = i+1
        createDataset(size, number, amount, source_dream, path, preprocessing)
    # GNW files
    createDataset(size, 1, 10, source_gnw, path, preprocessing)
    
create_all()
# preprocessing = 'KO'

# size = 10
# source = ('DREAM', '~', '~')
# number = 1
# path = r'C:\Users\Rien\CloudDiensten\Stack\Documenten\Python Scripts\BEP'
# KO, KD, WT, y = openFiles(size, number, source, path)
# dataset, x, rotated = dataPrep(KO, KD, WT, GS, preprocessing)
# rotated = np.roll(KO, -1, axis = 0)
# labels, GS = reshapeLabels(GS, size)
