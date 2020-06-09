import csv
import numpy as np
import time
start_time = time.time()


def loadData(file):
    ''' load the initial data, and calculate the mean of every gene over the 100 experiments'''
    tsv_file = open(file)
    read_tsv = csv.reader(tsv_file, delimiter = "\t")
    KOdata = np.array(list(read_tsv)[1:]).astype("float")
    tsv_file.close()
    return KOdata

def loadGSdata(file,size):
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

def dataprep(KOdata,WTdata,size):
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

def openFiles(size, number):
    cur_path = r'C:\\Users\\Rien\\CloudDiensten\\Stack\\Documenten\\Python Scripts\\gnw-master'
    new_path = cur_path + '\\10times50_1'
    name = '\\Yeast-' + str(number) 
    KOpath = new_path + name + '_knockouts.tsv'
    WTpath = new_path + name + '_wildtype.tsv'
    GSpath = new_path + name + '_goldstandard.tsv'
    
    KOdata = loadData(KOpath)
    WTdata = loadData(WTpath)
    GSdata = loadGSdata(GSpath, size)
    
    return KOdata, WTdata, GSdata

complete_dataset=np.array([])
labels = []

for i in range(10):
    size = 50
    number = i + 1
    KOdata, WTdata, GSdata = openFiles(size,number)
    dataset = dataprep(KOdata,WTdata,GSdata)
    complete_dataset =np.append(complete_dataset,dataset)
    for k in range(len(KOdata)):
        for l in range(len(KOdata)):
            labels.append(GSdata[l,k])
    print(i+1)
    print("--- %s seconds ---" % (time.time() - start_time))


length = 4*size+2
dataset = complete_dataset.reshape([size*size*10,length])

np.savetxt('dataset.txt', dataset)
np.savetxt('labels.txt', labels)

print("--- %s seconds ---" % (time.time() - start_time))