import csv
import numpy as np
from scipy import stats


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
        GSreshaped[x][y] = edge
    
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
    # calculate element-wise Z-score
    Zko = stats.zscore(KOdata, axis = 0, ddof = 0)
    PZko = stats.norm.cdf(Zko)
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

    return Zko, PZko, PD

path = r'C:\\Users\\Rien\\CloudDiensten\\Stack\\Documenten\\Python Scripts\\BEP'
size = 100
number = 1
amount = 1
source = 'DREAM'

KOdata, WTdata, GSdata = openFiles(size, number, source, path)
data = dataPrep(KOdata, WTdata, size)