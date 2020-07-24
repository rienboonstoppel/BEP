from tensorflow import keras
import numpy as np
import tensorflow.keras.backend as K

def get_f1(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val

def loadData(data_newpath, labels_newpath, model_newpath):
    dataset = np.loadtxt(data_newpath)
    labels = np.loadtxt(labels_newpath).astype(int)
    model = keras.models.load_model(model_newpath, custom_objects = {'get_f1': get_f1}) # tensorflow 2.x necessary 
    model.summary()
    return dataset, labels, model

def prepareEdgelist(size):
    edges = []
    for i in range(size):
        for j in range(size):
            source_node = i+1
            target_node = j+1
            edges.append((source_node, target_node))
    edges = np.array(edges)
        
    selfloops = []
    for i in range(size):
        index = i*size + i
        selfloops.append(index)
    return edges, selfloops

def rankedEdgelist(model, dataset):
    predicted_proba = model.predict_proba(dataset)
    # predicted_proba = predicted_proba.reshape([100,100])

    edges, selfloops = prepareEdgelist(size)
                
    unsorted_edges = np.hstack((edges, predicted_proba))
    unsorted_edges = np.delete(unsorted_edges,selfloops,axis=0)
    sorted_edges = unsorted_edges[np.argsort(unsorted_edges[:, 2])]
    sorted_edges_desc = np.flip(sorted_edges,axis=0)
    ranked_edgelist = []
    for i in range(len(sorted_edges_desc)):
        ranked_edgelist.append(('G' + str(int(sorted_edges_desc[i][0])), 'G' + str(int(sorted_edges_desc[i][1])), sorted_edges_desc[i][2]))
    return ranked_edgelist

def saveEdgelist(edgelist, save_name):
    with open(save_name, 'w') as file:
        file.write('\n'.join('%s\t%s\t%.7f' % x for x in edgelist))

size = 100
path = r'C:\Users\Rien\CloudDiensten\Stack\Documenten\Python Scripts\BEP\data\\'
data_name = 'DREAM_1_1_100'
model_path = r'C:\Users\Rien\CloudDiensten\Stack\Documenten\Python Scripts\BEP\saved_models\predictions\\'
model_name ='fold-5.hdf5'
data_newpath = path + data_name + '_Zmax-data.txt'
labels_newpath = path + data_name + '_Zmax-labels.txt'
model_newpath = model_path + model_name
    
dataset, labels, model = loadData(data_newpath, labels_newpath, model_newpath)

model.summary()

savename = data_name + '_rankededgelist.tsv'
ranked_edgelist = rankedEdgelist(model,dataset)
ranked_edgelist = rankedEdgelist(model,dataset)
saveEdgelist(ranked_edgelist, savename)