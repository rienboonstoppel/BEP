import numpy as np
import csv
# import time
        
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
    return KOrotated

def makeLabels(size):
    labels = np.zeros(size).astype('int')
    indices = range(1,size+1)
    indices_reverse = range(size, 0, -1)
    
    edges = []
    for i in range(1):
        for j in range(size):
            source_node = indices[indices[j]-indices_reverse[i]]
            target_node = indices[j]
            edges.append((source_node, target_node))
            
    for j in range(len(edges)):
        if edges[j] in GSdata:
            labels[j] = 1
    return labels

path = r'C:\\Users\\Rien\\CloudDiensten\\Stack\\Documenten\\Python Scripts\\BEP'
size = 100
number = 1
amount = 1
source = 'DREAM'
KOdata, WTdata, GSdata = openFiles(size, number, source, path)
KOrotated = rotate(KOdata)

labels = makeLabels(size)