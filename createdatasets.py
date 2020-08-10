import csv
import numpy as np
import time
from scipy import stats
import os

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

def dataPrep(KO, KD, WT, GS, preprocessing, rotation):
    ''' create dataset of one experiment
    preprocessing: indicate what data will be included in the dataset
        - for KO the diagonal will be substituted with the wildtype to avoid zeros
        - for Zmax the data will first be corrected using a logarithmic transformation, and then be combined into a Z-matrix of maximum values of KO and KD
    rotation: use a rotation of the data. No rotation corresponds with option 1 in my paper, rotation corresponds with option 3
    '''
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

    if rotation:    
        # rotate the complete matrix in order to train on the first column every time
        for j in range(len(KO)):
            if preprocessing == 'KO':
                rotated = np.roll(KO, -j, axis = 0)
            elif preprocessing == 'Zmax':
                rotated = np.roll(Zmax, -j, axis = 0)
            for i in range(len(KO)):
                dataset.extend(rotated[:,i].tolist())
    else:
        for j in range(len(KO)):
            for i in range(len(KO)):
                if preprocessing == 'KO':
                    dataset.extend(KO[:,i].tolist())
                    dataset.extend(KO[i,:].tolist())
                    dataset.extend(KO[:,j].tolist())
                    dataset.extend(KO[j,:].tolist())
                elif preprocessing == 'Zmax':
                    dataset.extend(Zmax[:,i].tolist())
                    dataset.extend(Zmax[i,:].tolist())
                    dataset.extend(Zmax[:,j].tolist())
        
            
    return dataset

def reshapeLabels(GS, size):
    '''create labels array from GS'''
    labels = np.reshape(GS,[size*size,1])
    return labels

def createDataset(size, number, amount, source, path, preprocessing, rotation):
    '''create dataset based on variables:
            size = dimensions of the network
            number = number of the network, when multiple networks are used, this is the starting number
            amount = amount of networks to be added in dataset
            source = GNW or DREAM
            path = working directory
            preprocessing = raw data or Z stats
        and saves it to a folder 'data'
    '''
    if not os.path.exists(path + '\data'):
        os.makedirs(path + '\data')

    if source[0] == 'DREAM':
        savename = 'data\\' + source[0] + '_' + str(number) + '_' + str(amount) + '_' + str(size) + '_' + preprocessing
    elif source[0] =='GNW':
        savename = 'data\\' + source[0] + '-' + source[1] + '-' + source[2] + '_' + str(amount) + '_' + str(size) + '_' + preprocessing
    completeDataset = np.array([])
    allLabels = np.array([])
    for i in range(amount):
        start_time = time.time()
        KO, KD, WT, GS = openFiles(size, number, source, path)
        dataset = dataPrep(KO, KD, WT, GS, preprocessing, rotation)
        labels = reshapeLabels(GS, size)
        completeDataset = np.append(completeDataset,dataset)
        allLabels = np.append(allLabels,labels)
        print('Loading network ' + str(number) + ' took %s seconds' % (time.time() - start_time))
        number += 1
    if rotation:
        columns = 1 * size
    else: columns = 4 * size
    rows = amount * size * size
    completeDataset = completeDataset.reshape([rows,columns])
    start_time = time.time()
    np.savetxt(savename + '-data.txt', completeDataset)
    np.savetxt(savename + '-labels.txt', allLabels, fmt='%d')
    print('Writing complete dataset took %s seconds' % (time.time() - start_time))

    return completeDataset, allLabels

def create_all():
    '''
    this functions creates all the necessary files to start the training on the keras model. This is based on 10 GRNs generated with GWN, and the provided 5 DREAM GRNs
    The data should be located in:
        - For DREAM4 size 100 challenge in (...path...)\source_data\DREAM4\size100
        - For GWN size 100, greedy neighbour selection, nonoise in (...path...)\source_data\GWN\size100\greedy\nonoise - goldstandard in ...\greedy\goldstandard
    '''
    path = r'C:\Users\Rien\CloudDiensten\Stack\Documenten\Python Scripts\BEP'
    size = 100
    source_dream = ('DREAM', '~', '~')
    source_gnw = ('GNW', 'greedy', 'nonoise')
    preprocessing = 'Zmax' #KO, Zmax
    rotation = True
    # DREAM files
    for i in range(5):
        amount = 1
        number = i+1
        createDataset(size, number, amount, source_dream, path, preprocessing, rotation)
    # GNW files
    createDataset(size, 1, 10, source_gnw, path, preprocessing, rotation)
    
create_all()
