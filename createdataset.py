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

def dataPrep(KOdata,WTdata,size):
    ''' create dataset of one experiment containing for every row in dataset row of gene i, column of gene i, row of gene j, column of gene j, wildtype i and wildtype j'''
    dataset = np.array([])
    for j in range(len(KOdata)):
        for i in range(len(KOdata)):
            dataset = np.append(dataset, KOdata[:,i])
            dataset = np.append(dataset, KOdata[i,:])
            dataset = np.append(dataset, KOdata[:,j])
            dataset = np.append(dataset, KOdata[j,:])
            dataset = np.append(dataset, WTdata[0,i])
            dataset = np.append(dataset, WTdata[0,j])
    return dataset

def openFiles(size, number, source):
    '''load all data of one experiment, depending of source Gene Net Weaver or DREAMdata'''
    if source =='GNW':
        cur_path = r'C:\\Users\\Rien\\CloudDiensten\\Stack\\Documenten\\Python Scripts\\gnw-master'
        new_path = cur_path + '\\10times50_1'
        name = '\\Yeast-' + str(number) 
        KOpath = new_path + name + '_knockouts.tsv'
        WTpath = new_path + name + '_wildtype.tsv'
        GSpath = new_path + name + '_goldstandard.tsv'

    if source == 'DREAM':
        cur_path = r'C:\Users\Rien\CloudDiensten\Stack\Documenten\Python Scripts\DREAMdata'
        name = 'insilico_size' + str(size) + '_' + str(number)
        KOpath = cur_path + '\\size' + str(size) + '\\DREAM4trainingdata\\' + name  + '\\' + name + '_knockouts.tsv'
        WTpath = cur_path + '\\size' + str(size) + '\\DREAM4trainingdata\\' + name  + '\\' + name + '_wildtype.tsv'
        GSpath = cur_path + '\\size' + str(size) + '\\DREAM4goldstandards\\' + name + '_goldstandard.tsv'
    
    KOdata = loadData(KOpath)
    WTdata = loadData(WTpath)
    GSdata = loadGSdata(GSpath, size)
    
    return KOdata, WTdata, GSdata
 
def singleExperiment(size, number,source):
    KOdata, WTdata, GSdata = openFiles(size,number,source)
    dataset = dataPrep(KOdata,WTdata,GSdata)
    
    labels = []
    for k in range(len(KOdata)):
        for l in range(len(KOdata)):
            labels.append(GSdata[l,k])
        
    length = 4*size+2
    dataset = dataset.reshape([size*size,length])
    
    return dataset, labels

def multipleExperiments(size,amount,source):
    completeDataset=np.array([])
    labels = []
   
    start_time = time.time()
    
    for i in range(amount):
        number = i + 1
        KOdata, WTdata, GSdata = openFiles(size,number,source)
        dataset = dataPrep(KOdata,WTdata,GSdata)
        completeDataset =np.append(completeDataset,dataset)
        for k in range(len(KOdata)):
            for l in range(len(KOdata)):
                labels.append(GSdata[l,k])
        print("--- %s seconds ---" % (time.time() - start_time))
        
    length = 4*size+2
    completeDataset = completeDataset.reshape([size*size*amount,length])
    
    print("--- %s seconds ---" % (time.time() - start_time))
    
    return completeDataset, labels

# DREAM_1_100_data,DREAM_1_100_labels = singleExperiment(100,1,'DREAM')
GNW_100_5_data,GNW_100_5_labels = multipleExperiments(100,5,'GNW')

np.savetxt('data\GNW_100_5_data.txt', GNW_100_5_data)
np.savetxt('data\GNW_100_5_labels.txt', GNW_100_5_labels)
