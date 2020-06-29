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
        new_path = path + '\\source_data\\GNW\\' + source[1] + '\\' 
        name = '\\Yeast-' + str(number) 
        KOpath = new_path + source[2] + '\\' + name + '_knockouts.tsv'
        KDpath = new_path + source[2] + '\\' + name + '_knockdowns.tsv'
        WTpath = new_path + source[2] + '\\' + name + '_wildtype.tsv'
        GSpath = new_path + 'goldstandard\\' + name + '_goldstandard.tsv'
    elif source[0] == 'DREAM':
        new_path = path + '\\source_data\\DREAM4'
        name = 'insilico_size' + str(size) + '_' + str(number)
        KOpath = new_path + '\\size' + str(size) + '\\DREAM4trainingdata\\' + name  + '\\' + name + '_knockouts.tsv'
        KDpath = new_path + '\\size' + str(size) + '\\DREAM4trainingdata\\' + name  + '\\' + name + '_knockdowns.tsv'
        WTpath = new_path + '\\size' + str(size) + '\\DREAM4trainingdata\\' + name  + '\\' + name + '_wildtype.tsv'
        GSpath = new_path + '\\size' + str(size) + '\\DREAM4goldstandards\\' + name + '_goldstandard.tsv'
    
    KO = loadData(KOpath)
    KD = loadData(KDpath)
    WT = loadData(WTpath)
    GS = loadGSdata(GSpath, size)
    
    return KO, KD, WT, GS

def dataPrep(KOdata, WTdata, size):
    ''' create dataset of one experiment containing for every row in dataset row of gene i, column of gene i, row of gene j, column of gene j, wildtype i and wildtype j'''
    dataset = []

    Zko = np.absolute(stats.zscore(KOdata, axis = 0, ddof = 0))
    PZko = 2 * (1 - stats.norm.cdf(Zko))
    Prod = PZko # .* PZkd;
    PD = np.zeros((size,size));
    for i in range(size):
        for j in range(size):
            if Prod[i,j] > 0:
                # CDF of product of two random variables uniformly distributed on the unit interval [0, 1]
                PD[i,j] = Prod[i,j] * (1 - np.log(Prod[i,j])); 
            else:
                # % log(0) = inf
                PD[i,j] = 0;

    # set diagonal to zero
    for i in range(len(KOdata)):
        PD[i,i] = 1
    
    for j in range(len(KOdata)):
        for i in range(len(KOdata)):
            dataset.extend(PD[:,i].tolist())
            dataset.extend(PD[i,:].tolist())
            dataset.extend(PD[:,j].tolist())
            dataset.extend(PD[j,:].tolist())
            
    return dataset

def reshapeLabels(GS):
    '''create labels array from GS'''
    labels = []
    for k in range(len(GS)):
        for l in range(len(GS)):
            labels.append(GS[l,k])
    return labels

def createDataset(size, number, amount, source, path, suffix):
    '''create dataset based on variables:
            size = dimensions of the network
            number = number of the network, when multiple networks are used, this is the starting number
            amount = amount of networks to be added in dataset
            source = GNW or DREAM
            path = working directory
    '''
    if source[0] == 'DREAM':
        savename = 'data\\' + source[0] + '_' + str(amount) + '_' + str(size) + '_' + suffix
    elif source[0] =='GNW':
        savename = 'data\\' + source[0] + '-' + source[1] + '-' + source[2] + '_' + str(amount) + '_' + str(size) + '_' + suffix
    completeDataset = np.array([])
    allLabels = np.array([])
    for i in range(amount):
        start_time = time.time()
        KO, KD, WT, GS = openFiles(size, number, source, path)
        dataset = dataPrep(KO, WT, size)
        labels = reshapeLabels(GS)
        completeDataset = np.append(completeDataset,dataset)
        allLabels = np.append(allLabels,labels)
        print('Loading network ' + str(number) + ' took %s seconds' % (time.time() - start_time))
        number += 1

    columns = 4 * size # + 2
    rows = amount * size * size
    completeDataset = completeDataset.reshape([rows,columns])
    start_time = time.time()
    np.savetxt(savename + '-data.txt', completeDataset)
    np.savetxt(savename + '-labels.txt', allLabels)
    print('Writing complete dataset took %s seconds' % (time.time() - start_time))

    return completeDataset, allLabels

path = r'C:\Users\Rien\CloudDiensten\Stack\Documenten\Python Scripts\BEP'
size = 100
number = 1
amount = 1
source = ('DREAM', 'greedy', 'nonoise')
suffix = 'PD'

completeDataset, allLabels = createDataset(size, number, amount, source, path, suffix)
# allLabels = np. reshape(allLabels, (10000,3))