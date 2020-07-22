from tensorflow import keras
import numpy as np
import tensorflow.keras.backend as K
import csv

def get_f1(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val

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

pred_path = r'C:\Users\Rien\CloudDiensten\Stack\Documenten\Python Scripts\BEP\data\\'
pred_name = 'DREAM_5_1_100'
pred_dataset = np.loadtxt(pred_path + pred_name + '_Zmax-data.txt')
pred_labels = np.loadtxt(pred_path + pred_name + '_Zmax-labels.txt').astype(int)

model_path = r'C:\Users\Rien\CloudDiensten\Stack\Documenten\Python Scripts\BEP\saved_models\predictions\\'
model_name = 'fold-2.hdf5'
model = keras.models.load_model(model_path+model_name, custom_objects = {'get_f1': get_f1}) # tensorflow 2.x necessary 
# model = keras.models.load_model('model.hdf5',custom_objects={'get_f1':get_f1}) # tensorflow 2.x necessary 

model.summary()

path = r'C:\Users\Rien\CloudDiensten\Stack\Documenten\Python Scripts\BEP'
size = 100
number = 1
name = 'insilico_size' + str(size) + '_' + str(number)
new_path = path + '\\source_data\\DREAM4'
GSpath = new_path + '\\size' + str(size) + '\\DREAM4goldstandards\\' + name + '_goldstandard.tsv'
GSdata = loadGSdata(GSpath, size)

predicted_labels = model.predict_proba(pred_dataset)
predicted_labels_reshaped = predicted_labels.reshape([100,100])
edges = []
for i in range(size):
    for j in range(size):
        source_node = j+1
        target_node = i+1
        edges.append((source_node, target_node))
        
indices = []
for i in range(size):
    index = i*size + i
    indices.append(index)
        
edges = np.array(edges)
unsorted_edges = np.hstack((edges, predicted_labels))
unsorted_edges = np.delete(unsorted_edges,indices,axis=0)
sorted_edges = unsorted_edges[np.argsort(unsorted_edges[:, 2])]
save_list = []
for i in range(len(sorted_edges)):
    save_list.append(('G' + str(int(sorted_edges[i][0])), 'G' + str(int(sorted_edges[i][1])), sorted_edges[i][2]))
print('Total predicted edges = ', str(predicted_labels.sum()))

with open('sorted_edges_dream5.tsv', 'w') as file:
    file.write('\n'.join('%s\t%s\t%.7f' % x for x in save_list))